"""Multi-category standalone generation sweep (category first).

Shells out to ``scripts/generate_standalone.py`` once per category at the final
cell target (``per_cell * waves``). Generation of category N+1 overlaps with
masking/blending of category N (the mask GPU stays busy while gen GPUs move on).
Smoke pre-flight still runs gen+mask together so a failure aborts before wave 1.
``--continue`` on the generator skips rows already on disk.

    python scripts/generate_standalone_sweep.py
    python scripts/generate_standalone_sweep.py --dry-run --categories cable,hazelnut
    python scripts/generate_standalone_sweep.py --smoke-only
    python scripts/generate_standalone_sweep.py --skip-smoke --categories cable
    python scripts/generate_standalone_sweep.py --gpus 1,2,3,4 --mask-gpu 2,3

Hard-fails before generating anything if any resolved category lacks
``prompts/standalone/<cat>.json``. Smoke pre-flight (default on) writes to a separate
tree so 5-step samples cannot poison ``--continue`` dedupe against the real corpus.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from synevad.synthesis.category_overrides import apply_loaded_overrides, load_override_map
from synevad.synthesis.crops import crop_plan
from synevad.synthesis.devices import parse_gpu_id_list, parse_gpu_ids
from synevad.synthesis.inputs import load_sources
from synevad.synthesis.quota import format_progress, read_all_manifests, shortfall_table
from synevad.synthesis.standalone_prompts import load_standalone_file
from synevad.synthesis.sweep import (
    category_run_plan,
    cells_from_standalone,
    discover_categories,
    generated_counts,
    missing_prompt_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path(__file__).with_name("generate_standalone.py")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default="synevad/synthesis/configs/standalone_sweep.yaml",
        help="YAML forwarded to the generator (default: synevad/synthesis/configs/standalone_sweep.yaml)",
    )
    ap.add_argument(
        "--sources-root",
        default="data/synevad/MVTec",
        help="root of per-category source trees (default: data/synevad/MVTec)",
    )
    ap.add_argument(
        "--source-subdir",
        default="source",
        help="subdir under each category that holds clean frames (default: source)",
    )
    ap.add_argument(
        "--prompts-dir",
        default="prompts/standalone",
        help="standalone prompt JSON dir (default: prompts/standalone)",
    )
    ap.add_argument(
        "--out-root",
        default=os.environ.get("SYNTHETIC_BENCH", "data/bench/mvtec"),
        help="corpus root; per-category images land in <out-root>/<cat>/images",
    )
    ap.add_argument(
        "--per-cell",
        type=int,
        default=5,
        help="target rows per (mode × severity) cell after wave 1 (default 10)",
    )
    ap.add_argument(
        "--waves",
        type=int,
        default=2,
        help="number of waves; wave W targets per-cell * W (default 2)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=10,
        help="cap sources per category (forwarded; default 10 for stable seeds)",
    )
    ap.add_argument(
        "--categories",
        default=None,
        help="comma-separated category restrict; default = every category under --sources-root",
    )
    ap.add_argument(
        "--run-tag",
        default="standalone",
        help="run-ids become <tag>_w1, <tag>_w2, ... (default: standalone)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="pass --dry-run through; no models load",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort on the first non-zero category exit (default: warn and continue)",
    )
    # Smoke pre-flight.
    ap.add_argument(
        "--smoke-out-root",
        default=os.environ.get("SYNTHETIC_BENCH", "data/bench/mvtec") + "_smoke",
        help="separate tree for smoke images (must not equal --out-root)",
    )
    ap.add_argument(
        "--smoke-steps",
        type=int,
        default=4,
        help="diffusion steps for smoke pre-flight (default 4; Klein 9B is 4-step distilled)",
    )
    ap.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip smoke pre-flight (e.g. when resuming an interrupted sweep)",
    )
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the smoke pre-flight and stop",
    )
    ap.add_argument(
        "--gpus",
        default=None,
        help="forwarded to the generator (comma-separated GPU ids)",
    )
    ap.add_argument(
        "--mask-gpu",
        default=None,
        help="forwarded to the generator (comma-separated GPU ids for masking/blending)",
    )
    ap.add_argument(
        "--mask-workers",
        type=int,
        default=None,
        help="forwarded to the generator (blend process count; default: one per --mask-gpu)",
    )
    ap.add_argument(
        "--embed-gpu",
        type=int,
        default=None,
        help="forwarded to the generator (GPU for Qwen3; default: first gen GPU)",
    )
    ap.add_argument(
        "--overrides",
        default=None,
        help="forwarded to the generator (per-category generation/blending/masks YAML)",
    )
    ap.add_argument(
        "--negatives",
        action="store_true",
        help="forwarded: generate negative controls for the same cells (use a separate --out-root)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="forwarded: JSONL spans + nvidia-smi under each category's <out>/profile",
    )
    ap.add_argument(
        "--profile-cuda",
        action="store_true",
        help="forwarded: cuda.synchronize around profile spans",
    )
    ap.add_argument(
        "--profile-torch",
        action="store_true",
        help="forwarded: Chrome-trace the first edit per gen worker (implies --profile)",
    )
    return ap.parse_args()


def can_overlap_mask(args: argparse.Namespace) -> bool:
    """True when a dedicated mask GPU can blend category N while others generate N+1."""
    if args.dry_run:
        return False
    ids = parse_gpu_ids(getattr(args, "gpus", None))
    if ids is None:
        return True
    mask_ids = (
        parse_gpu_id_list(args.mask_gpu)
        if getattr(args, "mask_gpu", None) is not None
        else [ids[-1]]
    )
    if not mask_ids:
        return False
    gen = [g for g in ids if g not in set(mask_ids)]
    return bool(gen)


def resolve_categories(args: argparse.Namespace) -> list[str]:
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        if not cats:
            raise SystemExit("--categories is empty")
        return cats
    return discover_categories(args.sources_root, args.source_subdir, args.prompts_dir)


def count_sources(sources_dir: Path, limit: int | None) -> int:
    """Number of source frames the generator will see (after --limit)."""
    return len(load_sources(str(sources_dir), limit=limit))


def crops_per_source(args: argparse.Namespace, category: str, sources_dir: Path) -> int:
    """How many crop windows the generator will cut from one frame of ``category``.

    Resolved the same way the generator resolves it — config ``cropping:`` block plus the
    category overlay — because the quota math is per ROI, not per frame, and a tiled
    category yields 5-6 ROIs from every frame. Frames of a category are uniform in size,
    so the first one settles it.
    """
    cfg = OmegaConf.load(str(args.config))
    overrides = load_override_map(
        cfg, config_path=args.config, cli_path=getattr(args, "overrides", None)
    )
    cfg, _ = apply_loaded_overrides(cfg, category, overrides)
    block = OmegaConf.to_container(cfg.get("cropping"), resolve=True) or {}
    plan = crop_plan(block if isinstance(block, dict) else {})
    first = load_sources(str(sources_dir), limit=1)[0]
    with Image.open(first) as im:
        size = im.size
    return len(plan.boxes(size))


def category_progress(
    cat: str,
    *,
    prompts_dir: Path,
    out_dir: Path,
    target: int,
) -> str:
    prompts = load_standalone_file(prompts_dir / f"{cat}.json")
    cells = cells_from_standalone(prompts)
    counts = generated_counts(read_all_manifests(out_dir))
    table = shortfall_table(cells, counts, target)
    return format_progress(table)


def pending_rows(
    cat: str,
    *,
    prompts_dir: Path,
    out_dir: Path,
    target: int,
) -> int:
    """How many rows are still short of ``target`` across all cells (sum of deficits)."""
    prompts = load_standalone_file(prompts_dir / f"{cat}.json")
    cells = cells_from_standalone(prompts)
    counts = generated_counts(read_all_manifests(out_dir))
    table = shortfall_table(cells, counts, target)
    return sum(max(0, need - have) for stages in table.values() for have, need in stages.values())


def build_command(
    args: argparse.Namespace,
    *,
    category: str,
    sources: Path,
    out: Path,
    num_per_source: int,
    run_id: str,
    limit: int,
    steps: int | None = None,
    dry_run: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--config",
        str(args.config),
        "--category",
        category,
        "--sources",
        str(sources),
        "--prompts-dir",
        str(args.prompts_dir),
        "--out",
        str(out),
        "--num-per-source",
        str(num_per_source),
        "--limit",
        str(limit),
        "--continue",
        "--run-id",
        run_id,
    ]
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if dry_run:
        cmd.append("--dry-run")
    if getattr(args, "gpus", None):
        cmd += ["--gpus", str(args.gpus)]
    if getattr(args, "mask_gpu", None) is not None:
        cmd += ["--mask-gpu", str(args.mask_gpu)]
    if getattr(args, "mask_workers", None) is not None:
        cmd += ["--mask-workers", str(args.mask_workers)]
    if getattr(args, "embed_gpu", None) is not None:
        cmd += ["--embed-gpu", str(args.embed_gpu)]
    if getattr(args, "overrides", None):
        cmd += ["--overrides", str(args.overrides)]
    if getattr(args, "negatives", False):
        cmd.append("--negatives")
    if getattr(args, "profile", False) or getattr(args, "profile_torch", False):
        cmd.append("--profile")
    if getattr(args, "profile_cuda", False):
        cmd.append("--profile-cuda")
    if getattr(args, "profile_torch", False):
        cmd.append("--profile-torch")
    if extra:
        cmd += list(extra)
    return cmd


def _cmd_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return env


def run_logged(cmd: list[str], log_path: Path | None) -> int:
    """Run ``cmd`` streaming stdout/stderr to the console and optionally tee to ``log_path``."""
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    if log_path is None:
        return subprocess.run(cmd, env=_cmd_env()).returncode
    handle = start_logged(cmd, log_path, announce=False)
    return handle.wait()


class LoggedProc:
    """Background subprocess whose stdout is pumped to the console and a log file."""

    def __init__(self, proc: subprocess.Popen, pump: threading.Thread):
        self.proc = proc
        self._pump = pump

    def wait(self) -> int:
        rc = self.proc.wait()
        self._pump.join()
        return rc


def start_logged(
    cmd: list[str], log_path: Path, *, announce: bool = True
) -> LoggedProc:
    """Start ``cmd`` without blocking; tee stdout/stderr until :meth:`LoggedProc.wait`.

    Reads the child pipe in binary chunks so tqdm ``\\r`` progress bars cannot fill
    the OS pipe and deadlock the blend workers while the next category generates.
    """
    import codecs

    if announce:
        print(f"[cmd] {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=_cmd_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdout is not None

    def pump() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    text = decoder.decode(b"", final=True)
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        log.write(text)
                    break
                text = decoder.decode(chunk)
                if not text:
                    continue
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text)
                log.flush()
        finally:
            log.close()

    thread = threading.Thread(target=pump, name="sweep-log-pump", daemon=False)
    thread.start()
    return LoggedProc(proc, thread)


def run_smoke(args: argparse.Namespace, categories: list[str]) -> None:
    """One image per (mode × severity) cell at low steps into the smoke tree.

    Any non-zero exit aborts the driver before wave 1, regardless of ``--fail-fast``.
    """
    smoke_root = Path(args.smoke_out_root)
    out_root = Path(args.out_root)
    if smoke_root.resolve() == out_root.resolve():
        raise SystemExit(
            "--smoke-out-root must differ from --out-root "
            "(smoke samples must not share --continue dedupe with the corpus)"
        )
    print(
        f"\n[smoke] pre-flight: {len(categories)} categor(ies), "
        f"steps={args.smoke_steps}, out={smoke_root}",
        flush=True,
    )
    for cat in categories:
        sources = Path(args.sources_root) / cat / args.source_subdir
        out = smoke_root / cat / "images"
        cmd = build_command(
            args,
            category=cat,
            sources=sources,
            out=out,
            num_per_source=1,
            run_id=f"{args.run_tag}_smoke",
            limit=1,
            steps=args.smoke_steps,
            dry_run=args.dry_run,
        )
        log = None if args.dry_run else out / "sweep_smoke.log"
        rc = run_logged(cmd, log)
        if rc != 0:
            raise SystemExit(
                f"[smoke] {cat} failed (exit {rc}); aborting before wave 1"
            )
        print(f"[smoke] {cat} ok", flush=True)
    print("[smoke] all categories passed pre-flight", flush=True)


def main() -> int:
    args = parse_args()
    if args.waves < 1:
        raise SystemExit("--waves must be >= 1")
    if args.per_cell < 1:
        raise SystemExit("--per-cell must be >= 1")
    if args.smoke_only and args.skip_smoke:
        raise SystemExit("pass only one of --smoke-only / --skip-smoke")

    categories = resolve_categories(args)
    missing = missing_prompt_files(categories, args.prompts_dir)
    if missing:
        raise SystemExit(
            "missing standalone prompt file(s) under "
            f"{args.prompts_dir}: {', '.join(missing)}. "
            "Derive them first, or restrict with --categories."
        )

    prompts_dir = Path(args.prompts_dir)
    out_root = Path(args.out_root)
    print(
        f"[sweep] {len(categories)} categor(ies): {', '.join(categories)}; "
        f"per-cell={args.per_cell}, waves={args.waves}, out={out_root}",
        flush=True,
    )

    if not args.skip_smoke:
        run_smoke(args, categories)
        if args.smoke_only:
            print("[done] --smoke-only; stopping before waves", flush=True)
            return 0

    n_sources_by_cat: dict[str, int] = {}
    n_crops_by_cat: dict[str, int] = {}
    any_failed = False
    for cat in categories:
        sources = Path(args.sources_root) / cat / args.source_subdir
        try:
            n_sources_by_cat[cat] = count_sources(sources, args.limit)
            n_crops_by_cat[cat] = crops_per_source(args, cat, sources)
        except SystemExit as e:
            print(f"[warn] {cat}: {e}", flush=True)
            n_sources_by_cat.pop(cat, None)
            any_failed = True
            if args.fail_fast:
                return 1

    plan = category_run_plan(
        [c for c in categories if c in n_sources_by_cat],
        per_cell=args.per_cell,
        waves=args.waves,
        n_rois_by_cat={
            cat: n * n_crops_by_cat[cat] for cat, n in n_sources_by_cat.items()
        },
        run_tag=args.run_tag,
    )
    target = args.per_cell * args.waves
    print(
        f"\n[plan] target {target} rows/cell"
        + (
            "; gen of category N+1 overlaps mask of N"
            if can_overlap_mask(args)
            else ""
        ),
        flush=True,
    )

    mask_handle: LoggedProc | None = None
    mask_cat: str | None = None

    def wait_mask() -> int:
        nonlocal mask_handle, mask_cat, any_failed
        if mask_handle is None:
            return 0
        cat = mask_cat or "?"
        rc = mask_handle.wait()
        mask_handle = None
        mask_cat = None
        if rc != 0:
            print(f"[warn] {cat} mask-only failed (exit {rc})", flush=True)
            any_failed = True
        else:
            print(
                f"[{cat}] progress (target {target}/cell):\n"
                + category_progress(
                    cat, prompts_dir=prompts_dir, out_dir=out_root / cat / "images",
                    target=target,
                ),
                flush=True,
            )
        return rc

    for entry in plan:
        cat = entry["category"]
        sources = Path(args.sources_root) / cat / args.source_subdir
        out = out_root / cat / "images"
        pending = pending_rows(
            cat, prompts_dir=prompts_dir, out_dir=out, target=entry["target"]
        )
        crops = n_crops_by_cat[cat]
        n_rois = n_sources_by_cat[cat] * crops
        print(
            f"\n[{cat}] {n_sources_by_cat[cat]} source(s) × {crops} crop(s) = "
            f"{n_rois} ROI(s), num_per_source={entry['num_per_source']}, "
            f"pending≈{pending}",
            flush=True,
        )
        # Every ROI gets every prompt: with more ROIs than the cell target there is no
        # attempt count below 1, so the overshoot is the ROI count itself. Say so, and
        # name the --limit that would land on target.
        if n_rois > entry["target"] and crops > 1:
            want = max(1, math.ceil(entry["target"] / crops))
            print(
                f"[{cat}] note: {n_rois} ROI(s) against a {entry['target']}-row target — "
                f"expect ≈{n_rois} rows/cell. Pass --limit {want} to land on target.",
                flush=True,
            )
        if not can_overlap_mask(args):
            cmd = build_command(
                args,
                category=cat,
                sources=sources,
                out=out,
                num_per_source=entry["num_per_source"],
                run_id=entry["run_id"],
                limit=args.limit,
                dry_run=args.dry_run,
            )
            log = None if args.dry_run else out / "sweep.log"
            rc = run_logged(cmd, log)
            if rc != 0:
                print(f"[warn] {cat} failed (exit {rc})", flush=True)
                any_failed = True
                if args.fail_fast:
                    wait_mask()
                    return 1
            elif not args.dry_run:
                print(
                    f"[{cat}] progress (target {entry['target']}/cell):\n"
                    + category_progress(
                        cat, prompts_dir=prompts_dir, out_dir=out, target=entry["target"]
                    ),
                    flush=True,
                )
            continue

        gen_cmd = build_command(
            args,
            category=cat,
            sources=sources,
            out=out,
            num_per_source=entry["num_per_source"],
            run_id=entry["run_id"],
            limit=args.limit,
            extra=["--skip-mask"],
        )
        rc = run_logged(gen_cmd, out / "sweep.log")
        if rc != 0:
            print(f"[warn] {cat} generate failed (exit {rc})", flush=True)
            any_failed = True
        prev_rc = wait_mask()
        if args.fail_fast and (rc != 0 or prev_rc != 0):
            return 1
        mask_cmd = build_command(
            args,
            category=cat,
            sources=sources,
            out=out,
            num_per_source=entry["num_per_source"],
            run_id=entry["run_id"],
            limit=args.limit,
            extra=["--mask-only"],
        )
        mask_handle = start_logged(mask_cmd, out / "sweep_mask.log")
        mask_cat = cat

    last_rc = wait_mask()
    if last_rc != 0 and args.fail_fast:
        return 1

    if any_failed:
        print("\n[done] sweep finished with failures", flush=True)
        return 1
    print("\n[done] sweep finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
