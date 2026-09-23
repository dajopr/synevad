"""Load standalone category prompt JSON (prompts/standalone/*.json).

Each file is a single JSON document with a flat ``generation_prompts`` list. Every
entry is an independent edit of a defect-free source (not a severity ladder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class StandalonePrompt:
    """One ready-to-send (mode, stage) generation instruction."""

    category: str
    mode: str
    stage: str
    prompt: str
    negatives: tuple[str, ...] = ()
    region_of_interest: str = ""
    # blob | line | cluster — the region a negative control standing in for this cell is
    # drawn as (synevad.synthesis.regions). Empty falls back to the config's `negatives.shapes`.
    region_shape: str = ""


def load_standalone_file(path: str | Path) -> list[StandalonePrompt]:
    """Parse one standalone category JSON into prompt records."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"standalone prompt file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "generation_prompts" not in data:
        raise SystemExit(
            f"{p}: expected a standalone prompt JSON with a 'generation_prompts' list"
        )
    category = str(data.get("category") or p.stem)
    out: list[StandalonePrompt] = []
    for i, item in enumerate(data["generation_prompts"]):
        if not isinstance(item, dict):
            raise SystemExit(f"{p}: generation_prompts[{i}] must be an object")
        mode = item.get("mode")
        stage = item.get("stage")
        prompt = item.get("prompt")
        if not mode or not stage or not prompt:
            raise SystemExit(
                f"{p}: generation_prompts[{i}] needs mode, stage, and prompt"
            )
        negatives = tuple(str(n) for n in (item.get("negatives") or []))
        out.append(
            StandalonePrompt(
                category=category,
                mode=str(mode),
                stage=str(stage),
                prompt=str(prompt),
                negatives=negatives,
                region_of_interest=str(item.get("region_of_interest") or ""),
                region_shape=str(item.get("region_shape") or ""),
            )
        )
    if not out:
        raise SystemExit(f"{p}: generation_prompts is empty")
    return out


def load_standalone_dir(
    prompts_dir: str | Path,
    *,
    category: str | None = None,
    modes: set[str] | None = None,
    stages: set[str] | None = None,
) -> list[StandalonePrompt]:
    """Load every ``*.json`` under ``prompts_dir``, optionally filtered."""
    root = Path(prompts_dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"prompts dir not found: {root}")
    files = sorted(root.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json prompt files in {root}")
    prompts: list[StandalonePrompt] = []
    for path in files:
        for rec in load_standalone_file(path):
            if category is not None and rec.category != category:
                continue
            if modes is not None and rec.mode not in modes:
                continue
            if stages is not None and rec.stage not in stages:
                continue
            prompts.append(rec)
    if not prompts:
        raise SystemExit(
            f"no prompts matched filters in {root}"
            + (f" (category={category!r})" if category else "")
        )
    return prompts
