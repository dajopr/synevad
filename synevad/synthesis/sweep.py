"""Planning helpers for the multi-category standalone generation sweep.

Everything here is pure stdlib (plus the standalone prompt loader) — no torch,
FLUX, or GPU — so the driver that shells out to ``generate_standalone.py`` can be
unit-tested on CPU. Reuses :mod:`synevad.synthesis.quota` for manifest reading and progress
tables; the ``{mode: {stage: (have, need)}}`` shape is the same as
``{mode: {severity: ...}}``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from synevad.synthesis.standalone_prompts import StandalonePrompt

# Canonical standalone severity order. Cells that never generated still appear as 0/N.
STAGE_ORDER = ("minimal", "slight", "moderate", "severe")


def discover_categories(
    sources_root: str | Path,
    source_subdir: str,
    prompts_dir: str | Path | None = None,
) -> list[str]:
    """Sorted category names under ``sources_root`` that contain ``<source_subdir>/``.

    ``prompts_dir`` is accepted for API symmetry with the driver but is not used for
    discovery — missing prompt files are reported separately via
    :func:`missing_prompt_files`.
    """
    del prompts_dir  # unused; kept so callers can pass the same trio of roots
    root = Path(sources_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"sources root not found: {root}")
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / source_subdir).is_dir():
            names.append(child.name)
    return names


def missing_prompt_files(
    categories: Sequence[str],
    prompts_dir: str | Path,
) -> list[str]:
    """Categories whose ``<cat>.json`` is absent under ``prompts_dir`` (sorted)."""
    root = Path(prompts_dir).expanduser()
    return sorted(c for c in categories if not (root / f"{c}.json").is_file())


def cells_from_standalone(
    prompts: Sequence[StandalonePrompt],
) -> dict[str, list[str]]:
    """Map each defect mode to its severity stages in canonical order.

    Stages that never appear in ``prompts`` are omitted; within a mode, stages
    follow ``STAGE_ORDER`` so a never-generated cell still shows as ``0/N`` once
    the mode is present. Modes keep first-seen order from ``prompts``.
    """
    by_mode: dict[str, set[str]] = {}
    mode_order: list[str] = []
    for p in prompts:
        if p.mode not in by_mode:
            by_mode[p.mode] = set()
            mode_order.append(p.mode)
        by_mode[p.mode].add(p.stage)
    out: dict[str, list[str]] = {}
    for mode in mode_order:
        stages = by_mode[mode]
        ordered = [s for s in STAGE_ORDER if s in stages]
        # Preserve any unexpected stage names after the canonical ones.
        ordered.extend(sorted(stages - set(STAGE_ORDER)))
        out[mode] = ordered
    return out


def generated_counts(rows: Iterable[dict]) -> dict[tuple[str, str], int]:
    """Count every persisted row per ``(class, severity)`` — scoring is off in the sweep.

    A negative control counts toward the defect cell it mirrors (``region_mode`` /
    ``region_stage``), so a negatives corpus fills the same cells as the defects it controls.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        cls = r.get("region_mode") or r.get("class")
        sev = r.get("region_stage") or r.get("severity")
        if cls is None or sev is None:
            continue
        counts[(str(cls), str(sev))] += 1
    return dict(counts)


def num_per_source_for(target_per_cell: int, n_rois: int) -> int:
    """How many attempts per ROI to reach ``target_per_cell`` rows in a cell.

    With 10 ROIs and target 10 this is 1 (wave 1); target 20 → 2 (wave 2).
    ``ceil`` keeps the driver honest if a category has fewer frames.

    ``n_rois`` is frames × crops-per-frame, not frames: ``--num-per-source`` is an
    attempt count *per crop*, so a cropped category (MVTec AD 2 tiles one frame into 5-6
    windows) already fills a cell several times over at one attempt. Counting frames here
    would over-generate by exactly the crop factor.
    """
    if n_rois < 1:
        raise ValueError(f"n_rois must be >= 1, got {n_rois}")
    if target_per_cell < 1:
        raise ValueError(f"target_per_cell must be >= 1, got {target_per_cell}")
    return max(1, math.ceil(target_per_cell / n_rois))


def category_run_plan(
    categories: Sequence[str],
    *,
    per_cell: int,
    waves: int,
    n_rois_by_cat: dict[str, int],
    run_tag: str,
) -> list[dict]:
    """One generator invocation per category at the final cell target.

    ``waves`` only scales the target (``per_cell * waves``); the driver no longer
    walks wave × category. Run ids are ``<run_tag>_<category>`` so stacked categories
    in one out-root never collide. ``n_rois_by_cat`` counts frames × crops-per-frame —
    see :func:`num_per_source_for`.
    """
    if waves < 1:
        raise ValueError(f"waves must be >= 1, got {waves}")
    if per_cell < 1:
        raise ValueError(f"per_cell must be >= 1, got {per_cell}")
    target = per_cell * waves
    plan: list[dict] = []
    for cat in categories:
        plan.append(
            {
                "category": cat,
                "target": target,
                "num_per_source": num_per_source_for(target, n_rois_by_cat[cat]),
                "run_id": f"{run_tag}_{cat}",
            }
        )
    return plan
