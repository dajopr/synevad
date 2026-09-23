"""Per-cell progress bookkeeping for the standalone generation sweep.

A **cell** is a ``(mode, severity)`` pair — one defect mode at one severity stage.
``scripts/generate_standalone_sweep.py`` tallies what is already on disk per cell and
prints how far each mode is from its target, so a resumed sweep only generates the
remainder.

Pure stdlib — no torch, FLUX, or model of any kind — so the orchestrator that shells out
to the generator can be unit-tested on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_all_manifests(out_dir: str | Path) -> list[dict]:
    """Read every ``generation_manifest_*.jsonl`` under ``out_dir`` into one list of rows.

    Progress is tallied from the per-wave manifests (the generator's canonical record), so
    any candidate already on disk — earlier waves, or a prior fixed-count run — counts toward
    the target and the sweep resumes rather than starting from zero.
    """
    rows: list[dict] = []
    for p in sorted(Path(out_dir).glob("generation_manifest_*.jsonl")):
        rows.extend(json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())
    return rows


def target_for(mode: str, target: int, overrides: dict[str, int]) -> int:
    """Per-mode quota: the override for this mode, else the global ``target``."""
    return overrides.get(mode, target)


def shortfall_table(
    cells: dict[str, list[str]],
    counts: dict[tuple[str, str], int],
    target: int,
    overrides: dict[str, int] | None = None,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Full ``{mode: {stage: (have, need)}}`` table (every stage, met or not).

    ``have`` is the count for the cell (0 if never generated); ``need`` is the mode's
    target. A mode is satisfied iff ``have >= need`` for every one of its stages.
    Callers use this both to decide the next wave and to print progress.
    """
    overrides = overrides or {}
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for mode, stages in cells.items():
        need = target_for(mode, target, overrides)
        table[mode] = {st: (counts.get((mode, st), 0), need) for st in stages}
    return table


def is_satisfied(stage_cells: dict[str, tuple[int, int]]) -> bool:
    """A mode is satisfied when every one of its cells has ``have >= need``."""
    return all(have >= need for have, need in stage_cells.values())


def format_progress(table: dict[str, dict[str, tuple[int, int]]]) -> str:
    """A compact per-mode progress block (one line per mode, cells + an ok/short marker)."""
    if not table:
        return "  (no modes)"
    width = max(len(mode) for mode in table)
    lines: list[str] = []
    for mode, cells in table.items():
        cell_str = "  ".join(f"{st} {have}/{need}" for st, (have, need) in cells.items())
        marker = "ok " if is_satisfied(cells) else "SHORT"
        lines.append(f"  [{marker}] {mode:<{width}}  {cell_str}")
    return "\n".join(lines)
