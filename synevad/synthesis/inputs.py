"""Load the clean source frames a generation run edits.

Pure stdlib — no model, cheap to import and CPU-testable. Prompts are loaded separately by
:mod:`synevad.synthesis.standalone_prompts`.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_sources(sources_dir: str, *, limit: int | None = None) -> list[Path]:
    """Sorted clean-source image paths under ``sources_dir`` (optionally capped at ``limit``).

    Sorted so a run is deterministic and ``limit`` always takes the same first N frames.
    Paths (not opened images) so the job builder can open each frame once, lazily.
    """
    root = Path(sources_dir)
    if not root.is_dir():
        raise SystemExit(f"sources is not a directory: {root}")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no source images found in {root}")
    return paths[:limit] if limit else paths
