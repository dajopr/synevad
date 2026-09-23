"""Write the generation manifest (one JSON object per candidate, JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

# Leftover HITL / chained columns. New rows never write these; scoring a corpus that
# still has them rewrites the verdict onto ``scorer`` and drops the rest.
_LEGACY_HITL_KEYS = (
    "vlm_decision",
    "vlm_score",
    "vlm_failed_criterion",
    "chain_id",
    "chain_step",
)


def apply_scorer(row: dict, decision: str | None) -> None:
    """Set the EditReward verdict (``accept`` / ``reject`` / null) and drop HITL leftovers."""
    row["scorer"] = decision
    for key in _LEGACY_HITL_KEYS:
        row.pop(key, None)


def write_manifest(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as JSON Lines (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_manifest(path: Path, rows: list[dict]) -> None:
    """Append ``rows`` to an existing JSONL (create the file if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_manifest(path: Path) -> list[dict]:
    """Read a JSONL manifest back into a list of rows (empty list if the file is absent)."""
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
