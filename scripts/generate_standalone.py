"""Minimal standalone MVTec defect generation for Synevad.

Driven by a YAML config (paths, generation, blend/mask, EditReward). Loads category
prompt JSONs from ``prompts_dir``, edits each clean source independently per
``(mode, stage)`` with FLUX.2 Klein, blends/masks via ``synevad.synthesis``, measures the
mask's area, scores with EditReward, and writes a Synevad-compatible
``generation_manifest_*.jsonl``.

Severity is the standalone stage name unchanged (``minimal`` / ``slight`` /
``moderate`` / ``severe``).

The ``cropping:`` block decides what "one edit" covers. ``crop_size: null`` (the MVTec AD
default) edits the whole frame; ``crop_size: N`` with ``layout: center | grid | centers``
edits square windows instead, which is what the much larger MVTec AD 2 frames need. Each
window is its own ROI (``<source>__crop<i>``), so ``--continue`` and the manifest keep
them apart.

Example::

    PYTHONPATH=$PWD python scripts/generate_standalone.py \\
      --config synevad/synthesis/configs/standalone_sweep.yaml

CLI flags only override selected config fields. Use ``--dry-run`` to print the
expanded job count without loading models; set ``scoring.enabled: false`` (or
``--no-score``) to skip EditReward. Use ``--continue`` (or ``continue_run: true`` in
the YAML) to skip every ``(roi, mode, stage, attempt)`` already present in any
``generation_manifest*.jsonl`` under ``out_dir`` and only generate the rest.
A raw edit already on disk (``crop_edited/``) with no manifest row is blended only.
Per-category overlays live in ``synevad/synthesis/configs/category_overrides.yaml`` (or ``--overrides``).

``--negatives`` (or ``negatives.enabled: true``) generates **negative controls** instead of
defects, for the same sources × crops × (mode, stage) cells: each cell's prompt is replaced by
``negatives.prompt`` (a clean, undamaged surface), a region shaped like that cell's defect is
drawn first (``synevad.synthesis.regions``), Klein inpaints only that region, and the result is pasted back
through the same blend as a defect. Rows are ``class: good`` with an empty mask. Write them to
their own ``--out``; EditReward is skipped for them. Klein only.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from synevad.data.synevad import scorer_verdict
from synevad.synthesis.category_overrides import apply_loaded_overrides, format_applied, load_override_map
from synevad.synthesis.crops import aspect_warning, crop_plan, parse_resize_to
from synevad.synthesis.devices import parse_device_id, parse_gpu_id_list, resolve_assignment
from synevad.synthesis.editreward import load_inferencer, score_margin
from synevad.synthesis.generate import normalize_quantization, uses_klein
from synevad.synthesis.inputs import IMAGE_EXTS, load_sources
from synevad.synthesis.manifest import apply_scorer, read_manifest, write_manifest
from synevad.synthesis.persist import artifact_dirs
from synevad.synthesis.standalone_jobs import (
    build_jobs,
    filter_completed,
    filter_completed_continue,
    partition_pending,
)
from synevad.synthesis.standalone_prompts import load_standalone_dir
from synevad.synthesis.standalone_run import StandaloneRun, free_gpu, run_jobs


def load_config(config_path: str) -> DictConfig:
    """Load the YAML config and resolve ``${oc.env:...}`` interpolations."""
    p = Path(config_path).expanduser()
    if not p.is_file():
        raise SystemExit(f"--config not found: {p}")
    cfg = OmegaConf.load(str(p))
    assert isinstance(cfg, DictConfig)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_sources(spec: str, *, limit: int | None = None) -> list[Path]:
    """Resolve a sources spec to sorted image paths (dir / file / glob)."""
    p = Path(str(spec)).expanduser()
    if p.is_dir():
        return load_sources(str(p), limit=limit)
    if p.is_file():
        return [p]
    matches = sorted(Path(m) for m in glob.glob(str(p)))
    imgs = [m for m in matches if m.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        raise SystemExit(f"sources matched no images: {spec!r}")
    return imgs[:limit] if limit else imgs


def parse_filter_set(raw) -> set[str] | None:
    """Accept null, a YAML list, or a comma-separated string as a filter set."""
    if raw is None:
        return None
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return set(items) or None
    text = str(raw).strip()
    if not text:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def _plain(node) -> dict:
    """OmegaConf block → plain dict (empty if the block is missing)."""
    if node is None:
        return {}
    out = OmegaConf.to_container(node, resolve=True)
    return dict(out) if isinstance(out, dict) else {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        required=True,
        help="YAML config (required), e.g. synevad/synthesis/configs/standalone_sweep.yaml",
    )
    ap.add_argument("--sources", default=None, help="override config.sources")
    ap.add_argument("--prompts-dir", default=None, help="override config.prompts_dir")
    ap.add_argument("--out", default=None, help="override config.out_dir")
    ap.add_argument("--category", default=None, help="override config.category")
    ap.add_argument("--modes", default=None, help="override config.modes (comma-separated)")
    ap.add_argument("--stages", default=None, help="override config.stages (comma-separated)")
    ap.add_argument("--limit", type=int, default=None, help="override config.limit")
    ap.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="override config.cropping.crop_size (square window; 0 = whole frame)",
    )
    ap.add_argument(
        "--crop-layout",
        default=None,
        help="override config.cropping.layout (full | center | grid | centers)",
    )
    ap.add_argument(
        "--crop-centers",
        default=None,
        help="override config.cropping.crop_centers as 'CX,CY;CX,CY' (implies layout centers)",
    )
    ap.add_argument(
        "--num-per-source",
        type=int,
        default=None,
        help="override config.generation.num_per_source",
    )
    ap.add_argument("--base-model", default=None, help="override config.generation.base_model")
    ap.add_argument("--seed", type=int, default=None, help="override config.generation.seed")
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override config.generation.steps (diffusion steps)",
    )
    ap.add_argument(
        "--quantization",
        default=None,
        help="override config.generation.quantization ('8bit' or 'none')",
    )
    ap.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids (e.g. 1,2,3,4). Last is the mask GPU unless --mask-gpu",
    )
    ap.add_argument(
        "--mask-gpu",
        default=None,
        help="GPU id(s) reserved for mask/blend workers (comma-separated, e.g. 3 or 2,3; "
        "default: last id in --gpus)",
    )
    ap.add_argument(
        "--mask-workers",
        type=int,
        default=None,
        help="number of blend processes (default: one per --mask-gpu id; "
        "use 2 with a single mask GPU to run two SAM workers on that card)",
    )
    ap.add_argument(
        "--embed-gpu",
        type=int,
        default=None,
        help="GPU id for Qwen3 prompt embeddings (default: first generation GPU; "
        "encoded inside that FLUX worker before it loads the DiT)",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="unique tag for filenames + manifest (default: timestamp); reuse to resume one run",
    )
    ap.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="skip (roi, mode, stage, attempt) already in any out_dir generation_manifest*; "
        "only generate missing jobs (overrides config.continue_run)",
    )
    ap.add_argument(
        "--skip-mask",
        action="store_true",
        help="write raw edits only; do not start the mask/blend worker",
    )
    ap.add_argument(
        "--mask-only",
        action="store_true",
        help="blend raw edits already on disk; do not run FLUX",
    )
    ap.add_argument(
        "--no-score",
        action="store_true",
        help="skip EditReward (overrides scoring.enabled)",
    )
    ap.add_argument(
        "--negatives",
        action="store_true",
        help="generate negative controls (inpainted clean regions) instead of defects; "
        "overrides negatives.enabled",
    )
    ap.add_argument(
        "--overrides",
        default="synevad/synthesis/configs/category_overrides.yaml",
        help="YAML of per-category generation/blending/masks overlays "
        "(default: config.overrides, e.g. synevad/synthesis/configs/category_overrides.yaml)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print expanded job count and exit without loading models",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="write JSONL spans + nvidia-smi samples under <out>/profile",
    )
    ap.add_argument(
        "--profile-cuda",
        action="store_true",
        help="with --profile, cuda.synchronize around spans (accurate GPU time, slower)",
    )
    ap.add_argument(
        "--profile-torch",
        action="store_true",
        help="Chrome-trace the first edit_image in each gen worker (implies --profile); "
        "captures aten::to / copy dtype conversions",
    )
    return ap.parse_args()


def apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace) -> None:
    if args.sources is not None:
        cfg.sources = args.sources
    if args.prompts_dir is not None:
        cfg.prompts_dir = args.prompts_dir
    if args.out is not None:
        cfg.out_dir = args.out
    if args.category is not None:
        cfg.category = args.category
    if args.modes is not None:
        cfg.modes = args.modes
    if args.stages is not None:
        cfg.stages = args.stages
    if args.limit is not None:
        cfg.limit = args.limit
    if any(
        v is not None for v in (args.crop_size, args.crop_layout, args.crop_centers)
    ) and ("cropping" not in cfg or cfg.cropping is None):
        cfg.cropping = OmegaConf.create({})
    if args.crop_size is not None:
        cfg.cropping.crop_size = args.crop_size or None
    if args.crop_layout is not None:
        cfg.cropping.layout = args.crop_layout
    if args.crop_centers is not None:
        cfg.cropping.crop_centers = [
            part.strip() for part in args.crop_centers.split(";") if part.strip()
        ]
        cfg.cropping.layout = "centers"
    if args.num_per_source is not None:
        cfg.generation.num_per_source = args.num_per_source
    if args.base_model is not None:
        cfg.generation.base_model = args.base_model
    if args.seed is not None:
        cfg.generation.seed = args.seed
    if args.steps is not None:
        cfg.generation.steps = args.steps
    if args.quantization is not None:
        cfg.generation.quantization = args.quantization
    if args.continue_run:
        cfg.continue_run = True
    if args.no_score:
        cfg.scoring.enabled = False
    if getattr(args, "negatives", False):
        if "negatives" not in cfg or cfg.negatives is None:
            cfg.negatives = OmegaConf.create({})
        cfg.negatives.enabled = True
    if args.gpus is not None:
        cfg.generation.devices = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.mask_gpu is not None:
        if "masks" not in cfg or cfg.masks is None:
            cfg.masks = OmegaConf.create({})
        ids = parse_gpu_id_list(args.mask_gpu)
        if not ids:
            raise SystemExit(f"invalid --mask-gpu {args.mask_gpu!r}")
        cfg.masks.device = f"cuda:{ids[0]}"
        if len(ids) > 1:
            cfg.masks.devices = [f"cuda:{i}" for i in ids]
    if getattr(args, "mask_workers", None) is not None:
        if "masks" not in cfg or cfg.masks is None:
            cfg.masks = OmegaConf.create({})
        cfg.masks.workers = args.mask_workers
    if args.embed_gpu is not None:
        cfg.generation.embed_device = f"cuda:{args.embed_gpu}"


def _maybe_activate_profile(args: argparse.Namespace, out_dir: Path) -> None:
    want = (
        bool(args.profile)
        or bool(getattr(args, "profile_torch", False))
        or bool(getattr(args, "profile_cuda", False))
        or bool(os.environ.get("SYNEVAD_PROFILE"))
        or bool(os.environ.get("SYNEVAD_PROFILE_TORCH"))
    )
    if not want:
        return
    from synevad.synthesis.profile import activate

    activate(
        out_dir / "profile",
        cuda=bool(getattr(args, "profile_cuda", False)),
        torch_trace=bool(getattr(args, "profile_torch", False))
        or bool(os.environ.get("SYNEVAD_PROFILE_TORCH")),
    )


def read_all_manifests(out_dir: Path) -> list[dict]:
    """Load every ``generation_manifest*`` JSONL under ``out_dir``."""
    paths = sorted(out_dir.glob("generation_manifest*"))
    rows: list[dict] = []
    for path in paths:
        if path.is_file():
            rows.extend(read_manifest(path))
    return rows


def negatives_block(cfg: DictConfig) -> dict | None:
    """The ``negatives:`` block as a plain dict when negative controls are on, else None."""
    block = _plain(cfg.get("negatives"))
    return block if block.get("enabled") else None


def negative_prompts(prompts, negatives: dict, *, base_model: str):
    """The defect cells with every prompt replaced by the negative-control prompt.

    Mode, stage and ``region_shape`` stay: they decide the region each negative is drawn as,
    and the stem / seed / resume key of the job, so the negatives corpus has exactly the cells
    of the defect corpus it controls.
    """
    if not uses_klein(base_model):
        raise SystemExit(f"--negatives needs a FLUX.2 Klein base model (inpainting); got {base_model}")
    text = " ".join(str(negatives.get("prompt") or "").split())
    if not text:
        raise SystemExit("negatives.prompt is empty — set the clean-surface prompt in the config")
    return [dataclasses.replace(p, prompt=text) for p in prompts]


def main() -> int:
    args = parse_args()
    if args.skip_mask and args.mask_only:
        raise SystemExit("pass only one of --skip-mask / --mask-only")
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    overrides = load_override_map(
        cfg, config_path=args.config, cli_path=args.overrides
    )
    cfg, overlay = apply_loaded_overrides(cfg, cfg.get("category"), overrides)
    apply_cli_overrides(cfg, args)  # CLI still wins over a category overlay
    if OmegaConf.to_container(overlay, resolve=True):
        print(format_applied(str(cfg.get("category") or ""), overlay), flush=True)

    missing = [
        key
        for key in ("sources", "prompts_dir", "out_dir")
        if not cfg.get(key)
    ]
    if missing:
        raise SystemExit(
            f"missing required config field(s): {', '.join(missing)} — set them in "
            f"{args.config} or override via CLI"
        )

    sources = resolve_sources(cfg.sources, limit=cfg.get("limit"))
    prompts = load_standalone_dir(
        cfg.prompts_dir,
        category=cfg.get("category"),
        modes=parse_filter_set(cfg.get("modes")),
        stages=parse_filter_set(cfg.get("stages")),
    )
    negatives = negatives_block(cfg)
    if negatives is not None:
        prompts = negative_prompts(prompts, negatives, base_model=str(cfg.generation.base_model))
    cropping = _plain(cfg.get("cropping"))
    plan = crop_plan(cropping)
    resize_to = parse_resize_to(cropping.get("resize_to"))
    jobs = build_jobs(
        sources,
        prompts,
        base_seed=int(cfg.generation.seed),
        num_per_source=int(cfg.generation.num_per_source),
        crops=plan,
    )
    # Frames of a category are uniform in size, so the first one's boxes describe the
    # plan; `jobs` is the authority on the total either way.
    with Image.open(sources[0]) as probe:
        first_size = probe.size
    n_crops = len(plan.boxes(first_size))
    print(
        f"[plan] {len(sources)} source(s) × {n_crops} crop(s) [{plan.describe()}] × "
        f"{len(prompts)} prompt(s) × {int(cfg.generation.num_per_source)} = "
        f"{len(jobs)} job(s) at {resize_to[0]}x{resize_to[1]}",
        flush=True,
    )
    warning = aspect_warning(first_size, plan.boxes(first_size)[0], resize_to)
    if warning:
        print(f"[plan] WARNING: {warning}", flush=True)

    out_dir = Path(cfg.out_dir)
    continue_run = bool(cfg.get("continue_run", False))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = out_dir / f"generation_manifest_{run_id}.jsonl"
    dirs = artifact_dirs(out_dir)

    if continue_run:
        prior_rows = read_all_manifests(out_dir) if out_dir.is_dir() else []
        pending_jobs, skipped = filter_completed_continue(jobs, prior_rows)
        manifest = read_manifest(manifest_path)
        n_manifests = (
            len(list(out_dir.glob("generation_manifest*"))) if out_dir.is_dir() else 0
        )
        print(
            f"[continue] {skipped} already on disk across {n_manifests} manifest(s); "
            f"{len(pending_jobs)} remaining",
            flush=True,
        )
    else:
        existing_rows = read_manifest(manifest_path)
        pending_jobs, manifest = filter_completed(jobs, existing_rows, run_id)
        if existing_rows:
            print(
                f"[resume] {len(existing_rows)} rows on disk -> "
                f"{len(pending_jobs)}/{len(jobs)} jobs pending",
                flush=True,
            )

    need_generate, blend_only = partition_pending(
        pending_jobs, run_id=run_id, dirs=dirs
    )
    if args.mask_only:
        need_generate = []
    if args.skip_mask:
        blend_only = []
    assignment = resolve_assignment(
        cli_gpus=args.gpus,
        cli_mask_gpu=args.mask_gpu,
        cli_embed_gpu=args.embed_gpu,
        cli_mask_workers=args.mask_workers,
        config_devices=cfg.generation.get("devices"),
        config_device=cfg.generation.get("device"),
        config_mask_device=(cfg.masks.get("device") if cfg.get("masks") else None),
        config_mask_devices=(cfg.masks.get("devices") if cfg.get("masks") else None),
        config_embed_device=cfg.generation.get("embed_device"),
        config_mask_workers=(cfg.masks.get("workers") if cfg.get("masks") else None),
    )

    if args.dry_run:
        by_mode = Counter(j.prompt.mode for j in pending_jobs)
        by_stage = Counter(j.prompt.stage for j in pending_jobs)
        by_cat = Counter(j.prompt.category for j in pending_jobs)
        print(f"[dry-run] pending jobs: {len(pending_jobs)}", flush=True)
        print(
            f"[dry-run] generate={len(need_generate)} blend-only={len(blend_only)}",
            flush=True,
        )
        print(f"[dry-run] categories: {dict(by_cat)}", flush=True)
        print(f"[dry-run] modes: {dict(by_mode)}", flush=True)
        print(f"[dry-run] stages: {dict(by_stage)}", flush=True)
        print(
            f"[dry-run] gpus gen={assignment.gen_gpus} mask={assignment.mask_gpus} "
            f"embed={assignment.embed_gpu} sequential={assignment.sequential}"
            f"{' skip-mask' if args.skip_mask else ''}"
            f"{' mask-only' if args.mask_only else ''}",
            flush=True,
        )
        return 0

    if not pending_jobs:
        if args.mask_only:
            print("[done] nothing to blend", flush=True)
        else:
            print("[done] nothing to generate", flush=True)
        return 0
    if args.mask_only and not blend_only:
        print("[done] nothing to blend", flush=True)
        return 0
    if args.skip_mask and not need_generate:
        print("[done] nothing to generate (blending deferred)", flush=True)
        return 0

    dirs.mkdir()
    _maybe_activate_profile(args, out_dir)
    print(
        f"[gpus] gen={assignment.gen_gpus} mask={assignment.mask_gpus} "
        f"embed={assignment.embed_gpu} sequential={assignment.sequential} "
        f"generate={len(need_generate)} blend-only={len(blend_only)}",
        flush=True,
    )

    masks = _plain(cfg.masks)
    if parse_device_id(masks.get("device")) is None:
        masks["device"] = assignment.mask_device
    run = StandaloneRun(
        run_id=run_id,
        dirs=dirs,
        manifest_path=manifest_path,
        resize_to=resize_to,
        steps=int(cfg.generation.steps),
        guidance=float(cfg.generation.guidance),
        base_model=str(cfg.generation.base_model),
        quantization=normalize_quantization(cfg.generation.get("quantization")),
        cpu_offload=bool(cfg.generation.get("cpu_offload", False)),
        blending=_plain(cfg.blending),
        masks=masks,
        negatives=negatives,
    )
    manifest = run_jobs(
        need_generate=need_generate,
        blend_only=blend_only,
        assignment=assignment,
        run=run,
        manifest=manifest,
        skip_mask=bool(args.skip_mask),
    )

    # --- EditReward margin scoring ------------------------------------------
    score_enabled = (
        bool(cfg.scoring.get("enabled", True)) and not args.skip_mask and negatives is None
    )
    if score_enabled:
        pending_rows = [r for r in manifest if scorer_verdict(r) is None]
        if pending_rows:
            print(
                f"[score] EditReward on {len(pending_rows)} unscored row(s) "
                f"(accept_margin={cfg.scoring.accept_margin})",
                flush=True,
            )
            inferencer = load_inferencer(
                checkpoint_path=str(cfg.scoring.editreward_checkpoint),
                config_path=str(cfg.scoring.editreward_config),
                device=str(cfg.scoring.get("device", assignment.mask_device)),
            )
            noop_cache: dict[tuple[str, str], float] = {}
            accept_margin = float(cfg.scoring.accept_margin)
            for row in pending_rows:
                clean = Image.open(row["clean_crop_path"]).convert("RGB")
                edit = Image.open(row["blended_patch_path"]).convert("RGB")
                verdict = score_margin(
                    inferencer,
                    clean,
                    edit,
                    row["prompt"],
                    noop_cache=noop_cache,
                    cache_key=(row["clean_crop_path"], row["prompt"]),
                    accept_margin=accept_margin,
                )
                apply_scorer(row, verdict.decision)
                row["reward_score"] = verdict.margin
                row["reward_edit_score"] = verdict.edit_score
                row["reward_noop_score"] = verdict.noop_score
            del inferencer
            free_gpu()
        write_manifest(manifest_path, manifest)
    else:
        reason = (
            "--skip-mask; no blended patches yet"
            if args.skip_mask
            else "negative controls carry no edit instruction to judge"
            if negatives is not None
            else "scoring.enabled=false"
        )
        print(f"[score] skipped ({reason})", flush=True)
        write_manifest(manifest_path, manifest)

    if not score_enabled:
        summary = f"{len(manifest)} candidates (unscored)"
    else:
        accepts = sum(scorer_verdict(r) == "accept" for r in manifest)
        summary = (
            f"{len(manifest)} candidates "
            f"({accepts} accept / {len(manifest) - accepts} reject)"
        )
    print(f"[done] {summary} -> {manifest_path.resolve()}", flush=True)

    if negatives is not None:
        changes = [float(r["region_change"]) for r in manifest if r.get("region_change") is not None]
        if changes:
            weak = sum(c < float(negatives.get("min_change", 1.0)) for c in changes)
            print(
                f"[negatives] {len(changes)} composited; region_change median "
                f"{statistics.median(changes):.2f} levels; {weak} below "
                f"negatives.min_change={negatives.get('min_change', 1.0)} (inpaint barely changed the region)",
                flush=True,
            )
    statuses = Counter(r.get("mask_status") for r in manifest)
    usable = statuses.get("ok", 0)
    if negatives is None:
        print(
            f"[masks] {usable}/{len(manifest)} usable as pixel GT — "
            + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items(), key=lambda kv: str(kv[0]))),
            flush=True,
        )
    if args.profile or args.profile_torch or os.environ.get("SYNEVAD_PROFILE"):
        from synevad.synthesis.profile import profile_dir, summarize

        dest = profile_dir()
        if dest is not None:
            print(summarize(dest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
