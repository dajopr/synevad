"""Post-hoc EditReward scoring for generation manifests — no FLUX, no regeneration.

Every candidate already on disk carries what the judge needs: the clean crop
(``crop_original/``), the composited crop (``crop_blended/``), and the instruction that
produced it. This script walks a corpus of ``generation_manifest*.jsonl``, scores the rows
that carry no verdict yet, and writes the margin back into the manifests it read.

Why scoring is a separate pass
------------------------------
The generator can score inline (``scoring.enabled: true``), but the sweep
(``scripts/generate_standalone_sweep.py``) deliberately does not:

* **The sweep's fill logic ignores decisions.** ``synevad.synthesis.sweep.generated_counts``
  counts every persisted row, so a cell reaches its target counting rejects either way.
  An inline verdict would drive nothing; making it drive
  something means generate-until-accepted, which needs unbounded attempts and breaks the
  deterministic ``attempt_index`` the waves depend on.
* **Inline scoring only ever covers part of the corpus.** In continue mode the generator holds
  just the current run-id's manifest in memory, so rows written by earlier runs keep
  ``scorer: null`` forever. This pass sees every manifest under ``--root`` and scores
  them all against one checkpoint and one margin.
* **One judge residency instead of one per category.** A 7B VL model loads once here, not once
  per (wave x category) subprocess — and a judge failure can no longer turn a category whose
  images all generated fine into a failed generation run.
* **The threshold is re-tunable offline.** The raw ``reward_edit_score``/``reward_noop_score``
  and their margin are persisted, so ``--rethreshold`` re-derives every decision at a new
  ``--accept-margin`` without touching the GPU.

Rows are scored regardless of ``mask_status``: the judge rates the *image*, and mask quality
is a separate axis the downstream filter applies on its own.

Usage
-----
::

    # what would be scored, plus the margin distribution of whatever is already scored
    python scripts/score_manifests.py --root /data/.../bench/v2/mvtec --dry-run

    # score every unscored row in the corpus (one model load, resumable)
    python scripts/score_manifests.py --root /data/.../bench/v2/mvtec

    # try the judge on 20 rows of one category first
    python scripts/score_manifests.py --root /data/.../bench/v2/mvtec \
        --categories cable --limit 20

    # re-derive decisions at a different threshold; no GPU, no model
    python scripts/score_manifests.py --root /data/.../bench/v2/mvtec \
        --rethreshold --accept-margin 0.25

The pass is resumable and idempotent: only rows with ``scorer is None`` are scored
(unless ``--rescore``), manifests are rewritten atomically, and a run interrupted halfway
leaves the rows it finished scored and the rest untouched.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from PIL import Image

from synevad.data.synevad import scorer_verdict
from synevad.synthesis.manifest import apply_scorer, read_manifest, write_manifest
from synevad.synthesis.sweep import STAGE_ORDER

# Defaults mirroring the `scoring` block of synevad/synthesis/configs/standalone*.yaml, so the script is
# usable without a --config.
DEFAULT_SCORING = {
    "editreward_checkpoint": os.environ.get("EDITREWARD_CHECKPOINT", ""),
    "editreward_config": os.environ.get("EDITREWARD_CONFIG", ""),
    "accept_margin": 0.0,
    "device": "cuda:0",
}

# A re-masking pass copies the manifests it replaces into <run-dir>/remask_backup/. Those are
# snapshots of rows that live elsewhere; scoring them would double-count the corpus.
BACKUP_DIR_SUFFIX = "_backup"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        required=True,
        help="corpus root (<root>/<cat>/images/...), a single run's out_dir, or one manifest file",
    )
    ap.add_argument(
        "--categories",
        default=None,
        help="comma-separated restrict, matched against each row's `category` field",
    )
    ap.add_argument(
        "--accept-margin",
        type=float,
        default=None,
        help=f"accept when margin >= this (default: the config's, else {DEFAULT_SCORING['accept_margin']})",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="YAML whose `scoring` block supplies checkpoint/config/device/accept_margin",
    )
    ap.add_argument("--checkpoint", default=None, help="override scoring.editreward_checkpoint")
    ap.add_argument(
        "--editreward-config", default=None, help="override scoring.editreward_config"
    )
    ap.add_argument("--device", default=None, help="override scoring.device (default cuda:0)")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-score rows that already carry a decision (e.g. a new checkpoint)",
    )
    ap.add_argument(
        "--rethreshold",
        action="store_true",
        help="no GPU: re-derive every decision from the stored reward_score at --accept-margin",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be scored and summarise existing margins; write nothing",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score at most N rows this pass; safe to interrupt — the rest stay unscored "
        "and are picked up by the next run",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="rewrite the manifest every N scored rows, so a crash keeps the work (default 50)",
    )
    return ap.parse_args()


def resolve_scoring(args: argparse.Namespace) -> dict:
    """Merge the built-in defaults, an optional config's ``scoring`` block, and CLI overrides."""
    cfg = dict(DEFAULT_SCORING)
    if args.config:
        from omegaconf import OmegaConf

        loaded = OmegaConf.load(str(Path(args.config).expanduser()))
        block = OmegaConf.to_container(loaded.get("scoring", {}) or {})
        for key in cfg:
            if block.get(key) is not None:
                cfg[key] = block[key]
    for key, value in (
        ("editreward_checkpoint", args.checkpoint),
        ("editreward_config", args.editreward_config),
        ("device", args.device),
        ("accept_margin", args.accept_margin),
    ):
        if value is not None:
            cfg[key] = value
    cfg["accept_margin"] = float(cfg["accept_margin"])
    return cfg


def discover_manifests(root: str | Path) -> list[Path]:
    """Every ``generation_manifest*.jsonl`` under ``root``, backups excluded.

    ``root`` may be a single manifest file, one run's ``out_dir``, or a corpus root holding
    ``<category>/images/`` trees — the recursive glob covers all three, so the same command
    scores one run or the whole benchmark.
    """
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"--root not found: {root}")
    paths = [
        p
        for p in sorted(root.rglob("generation_manifest*.jsonl"))
        if not any(part.endswith(BACKUP_DIR_SUFFIX) for part in p.relative_to(root).parts[:-1])
    ]
    if not paths:
        raise SystemExit(f"no generation_manifest*.jsonl under {root}")
    return paths


def parse_categories(spec: str | None) -> set[str] | None:
    cats = {c.strip() for c in (spec or "").split(",") if c.strip()}
    return cats or None


def select_rows(
    rows: Iterable[dict], *, categories: set[str] | None = None, rescore: bool = False
) -> list[dict]:
    """The rows this pass should send to the judge (the dicts themselves, not copies)."""
    out = []
    for row in rows:
        if categories is not None and str(row.get("category")) not in categories:
            continue
        if rescore or scorer_verdict(row) is None:
            out.append(row)
    return out


def apply_threshold(rows: Iterable[dict], accept_margin: float) -> tuple[int, int]:
    """Re-derive ``scorer`` from each row's stored margin. Returns ``(changed, unscored)``.

    A row with no ``reward_score`` has never met the judge, so it is left null rather than
    silently rejected — it needs a scoring pass, not a threshold.
    """
    changed = unscored = 0
    for row in rows:
        margin = row.get("reward_score")
        if margin is None:
            unscored += 1
            continue
        decision = "accept" if float(margin) >= accept_margin else "reject"
        if scorer_verdict(row) != decision:
            changed += 1
        apply_scorer(row, decision)
        row["reward_accept_margin"] = accept_margin
    return changed, unscored


def write_manifest_atomic(path: Path, rows: list[dict]) -> None:
    """Rewrite ``path`` via a temp file + rename, so an interrupted write cannot truncate it."""
    tmp = path.with_name(path.name + ".tmp")
    write_manifest(tmp, rows)
    os.replace(tmp, path)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 1]) of an already-sorted sequence."""
    if not values:
        return float("nan")
    idx = q * (len(values) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (idx - lo))


def accept_rate(margins: Sequence[float], threshold: float) -> float:
    return sum(m >= threshold for m in margins) / len(margins) if margins else float("nan")


def stage_sort_key(stage: str) -> tuple[int, str]:
    """Canonical severity order where known (standalone grades), alphabetical otherwise."""
    return (STAGE_ORDER.index(stage), "") if stage in STAGE_ORDER else (len(STAGE_ORDER), stage)


def margin_report(rows: Iterable[dict], accept_margin: float, *, ladder_steps: int = 7) -> str:
    """Margin distribution + the accept rate it implies, over rows that carry a score.

    The ladder is spread over the observed p5–p95 rather than a fixed set of thresholds,
    because the margin's scale depends on the checkpoint — it is the curve you pick
    ``--accept-margin`` off, and ``--rethreshold`` then applies that choice for free.
    """
    scored = [r for r in rows if r.get("reward_score") is not None]
    if not scored:
        return "  (no rows carry a reward_score yet)"
    margins = sorted(float(r["reward_score"]) for r in scored)
    lines = [
        f"  {len(margins)} scored row(s); accept_margin={accept_margin:g} -> "
        f"{accept_rate(margins, accept_margin):.1%} accept",
        "  quantiles   "
        + "  ".join(
            f"p{int(q * 100)} {percentile(margins, q):+.3f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        ),
    ]
    lo, hi = percentile(margins, 0.05), percentile(margins, 0.95)
    if hi > lo:
        ladder = [lo + (hi - lo) * i / (ladder_steps - 1) for i in range(ladder_steps)]
        lines.append(
            "  accept rate "
            + "  ".join(f"{t:+.2f}:{accept_rate(margins, t):.0%}" for t in ladder)
        )

    by_severity: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_severity[str(r.get("severity"))].append(float(r["reward_score"]))
        by_category[str(r.get("category"))].append(float(r["reward_score"]))

    # Severity is the one axis with a prior: a severe edit should out-score a minimal one.
    # A flat or inverted column here says the judge is not tracking severity on this corpus.
    def block(title: str, groups: dict[str, list[float]], keys: list[str]) -> list[str]:
        width = max((len(k) for k in keys), default=0)
        return [f"  by {title}"] + [
            f"    {k:<{width}}  n={len(groups[k]):<5} med {percentile(sorted(groups[k]), 0.5):+.3f}"
            f"  acc {accept_rate(groups[k], accept_margin):.1%}"
            for k in keys
        ]

    lines += block("severity", by_severity, sorted(by_severity, key=stage_sort_key))
    if len(by_category) > 1:
        lines += block("category", by_category, sorted(by_category))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.rethreshold and args.rescore:
        raise SystemExit("--rethreshold re-uses stored margins; --rescore recomputes them — pick one")
    if args.flush_every < 1:
        raise SystemExit("--flush-every must be >= 1")

    scoring = resolve_scoring(args)
    accept_margin = scoring["accept_margin"]
    categories = parse_categories(args.categories)
    manifests = discover_manifests(args.root)

    files = [(path, read_manifest(path)) for path in manifests]
    all_rows = [r for _, rows in files for r in rows]
    if categories is not None:
        all_rows = [r for r in all_rows if str(r.get("category")) in categories]
    pending = (
        {}
        if args.rethreshold
        else {
            id(r): r
            for _, rows in files
            for r in select_rows(rows, categories=categories, rescore=args.rescore)
        }
    )
    print(f"[score] {len(manifests)} manifest(s), {len(all_rows)} row(s) under {args.root}", flush=True)
    if not args.rethreshold:
        print(
            f"[score] {len(pending)} row(s) to score"
            + (" (--rescore: including already-scored rows)" if args.rescore else ""),
            flush=True,
        )

    # --- --rethreshold: stored margins only, no model, no images ------------------
    if args.rethreshold:
        changed = unscored = 0
        for path, rows in files:
            targets = rows if categories is None else [
                r for r in rows if str(r.get("category")) in categories
            ]
            c, u = apply_threshold(targets, accept_margin)
            changed += c
            unscored += u
            if c and not args.dry_run:
                write_manifest_atomic(path, rows)
        verb = "would change" if args.dry_run else "changed"
        print(
            f"[rethreshold] accept_margin={accept_margin:g}: {verb} {changed} decision(s); "
            f"{unscored} row(s) have no reward_score (run without --rethreshold to score them)\n"
            + margin_report(all_rows, accept_margin),
            flush=True,
        )
        if args.dry_run:
            print("[score] dry run — no files written", flush=True)
        return 0

    if args.dry_run:
        print(
            f"[dry-run] would load {scoring['editreward_checkpoint']} on {scoring['device']}\n"
            f"[dry-run] margins already on disk:\n" + margin_report(all_rows, accept_margin),
            flush=True,
        )
        return 0

    if not pending:
        print("[done] nothing to score", flush=True)
        return 0

    # --- scoring pass -------------------------------------------------------------
    from synevad.synthesis.editreward import load_inferencer, score_margin

    print(
        f"[score] loading EditReward: {scoring['editreward_checkpoint']} on {scoring['device']}",
        flush=True,
    )
    inferencer = load_inferencer(
        checkpoint_path=str(scoring["editreward_checkpoint"]),
        config_path=str(scoring["editreward_config"]),
        device=str(scoring["device"]),
    )
    # The no-op baseline score(clean, clean, prompt) depends only on the crop and the
    # instruction, so every attempt of one (source, mode, severity) cell shares it.
    noop_cache: dict[tuple[str, str], float] = {}
    checkpoint_name = Path(str(scoring["editreward_checkpoint"])).name

    total = len(pending) if args.limit is None else min(args.limit, len(pending))
    scored = skipped = 0
    started = time.time()
    stop = False

    for path, rows in files:
        targets = [r for r in rows if id(r) in pending]
        if not targets:
            continue
        dirty = 0
        try:
            for row in targets:
                if args.limit is not None and scored >= args.limit:
                    stop = True
                    break
                clean_path, edit_path = row.get("clean_crop_path"), row.get("blended_patch_path")
                if not (clean_path and edit_path and row.get("prompt")):
                    print(f"[warn] {row.get('image_id')}: missing crop path or prompt", flush=True)
                    skipped += 1
                    continue
                if not (Path(clean_path).exists() and Path(edit_path).exists()):
                    print(f"[warn] {row.get('image_id')}: crop file missing on disk", flush=True)
                    skipped += 1
                    continue

                clean = Image.open(clean_path).convert("RGB")
                edit = Image.open(edit_path).convert("RGB")
                verdict = score_margin(
                    inferencer,
                    clean,
                    edit,
                    row["prompt"],
                    noop_cache=noop_cache,
                    cache_key=(clean_path, row["prompt"]),
                    accept_margin=accept_margin,
                )
                apply_scorer(row, verdict.decision)
                row["reward_score"] = verdict.margin
                row["reward_edit_score"] = verdict.edit_score
                row["reward_noop_score"] = verdict.noop_score
                # Provenance: which judge and which threshold produced this decision, so a
                # corpus scored across several passes stays auditable.
                row["reward_model"] = checkpoint_name
                row["reward_accept_margin"] = accept_margin
                row["reward_scored_at"] = datetime.now().isoformat(timespec="seconds")

                scored += 1
                dirty += 1
                if dirty >= args.flush_every:
                    write_manifest_atomic(path, rows)
                    dirty = 0
                if scored % 25 == 0 or scored == total:
                    rate = scored / max(time.time() - started, 1e-9)
                    eta = (total - scored) / rate if rate else float("inf")
                    print(
                        f"[score] {scored}/{total} rows  {rate:.2f} row/s  eta {eta / 60:.1f} min",
                        flush=True,
                    )
        finally:
            # Runs on --limit, on an exception, and on Ctrl-C: never drop finished work.
            if dirty:
                write_manifest_atomic(path, rows)
        if stop:
            break

    del inferencer

    print(
        f"\n[done] scored {scored} row(s)"
        + (f", skipped {skipped} unreadable row(s)" if skipped else "")
        + f" in {(time.time() - started) / 60:.1f} min\n"
        + margin_report(all_rows, accept_margin)
        + "\n[hint] change the threshold without the GPU: "
        "--rethreshold --accept-margin <m>",
        flush=True,
    )
    remaining = len(pending) - scored - skipped
    if remaining > 0:
        print(f"[score] {remaining} row(s) still unscored — re-run to continue", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
