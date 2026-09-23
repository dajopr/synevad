"""Size negative-control regions from the defects they control.

A negative is matched to a defect cell by shape and size (``synevad.synthesis.regions``), and the size has
to come from somewhere. The defaults are guesses at a 1024 px crop, and a guess is easy to get
badly wrong: on the first synpic smoke run the stone-strike negatives covered 0.2-0.7% of the
crop where the defects FLUX actually drew covered 3-7%. A detector could then tell negatives
from defects by the size of the regenerated area alone.

This reads a generated **defect** corpus, takes the median extent of the estimated masks per
``(class, stage)`` over rows whose mask is a localised change (``ok`` and ``under_coverage`` by
default — ``over_coverage`` is FLUX re-rendering, not the defect), and converts it into the
region parameters of that class's shape, keeping the shape's other parameters at their rung:

* ``blob``    — ``radius = sqrt(area / pi)``, from the median mask area
* ``line``    — ``length`` = median **skeleton length** of the mask, read from the mask PNG.
  Not area / width: an estimated crack mask is several times wider than the drawn crack,
  so area / width reports a crack many crops long
* ``cluster`` — ``count = area / (pi * radius^2)``

Values are capped (a blob at a third of the crop edge, a line at 1.5 crop edges) so one
re-render mask swept into a cell cannot size every negative in it.

It prints a ``negatives.mode_sizes`` block to paste into the generation config, plus the
defect area next to the region area the current config would draw. Cells with fewer than
``--min-samples`` usable masks are left out and reported, so a thin corpus cannot set sizes
from one or two draws.

``--write-overrides PATH --categories a,b`` also writes those sizes as a per-category overrides
YAML that ``generate_standalone(_sweep).py --overrides PATH`` merges on top of the config, so a
scripted run can size its negatives without a hand edit. It is written even when no cell
qualifies (an empty overlay), leaving the config's sizes in force.

Usage::

    python scripts/calibrate_negative_sizes.py --root data/bench/cropped
    python scripts/calibrate_negative_sizes.py --root DEFECTS --write-overrides sizes.yaml \
        --categories pelton_cutout,pelton_face
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from synevad.synthesis.regions import REFERENCE_EDGE, region_spec, resolve_shape  # noqa: E402
from synevad.synthesis.sweep import STAGE_ORDER  # noqa: E402

DEFAULT_CONFIG = "synevad/synthesis/configs/cropped_example.yaml"
MAX_BLOB_RADIUS_FRAC = 1.0 / 3.0
MAX_LINE_LENGTH_FRAC = 1.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="defect corpus root (<root>/<cat>/images/generation_manifest*)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"generation config (default: {DEFAULT_CONFIG})")
    ap.add_argument("--statuses", default="ok,under_coverage", help="mask statuses that count as the defect")
    ap.add_argument("--min-samples", type=int, default=5, help="usable masks a cell needs (default 5)")
    ap.add_argument("--edge", type=int, default=REFERENCE_EDGE, help="crop edge the coverages are fractions of")
    ap.add_argument("--write-overrides", default=None, help="also write the sizes as a per-category overrides YAML")
    ap.add_argument("--categories", default=None, help="categories the overrides YAML applies to (comma-separated)")
    args = ap.parse_args(argv)
    if args.write_overrides and not args.categories:
        ap.error("--write-overrides needs --categories")
    return args


def read_rows(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(Path(root) / "*" / "images" / "generation_manifest*.jsonl"))):
        rows += [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows


def usable_rows(rows: list[dict], statuses: set[str]) -> dict[tuple[str, str], list[dict]]:
    """Defect rows with a non-empty mask of an accepted status, per (class, stage)."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("negative") or row.get("mask_status") not in statuses:
            continue
        if float(row.get("mask_coverage") or 0.0) > 0:
            cells[(str(row["class"]), str(row["severity"]))].append(row)
    return cells


def skeleton_length(mask_path: str | Path, edge: int) -> float:
    """Skeleton length of a binary mask in px at ``edge`` (diagonal steps count sqrt 2)."""
    from PIL import Image

    img = Image.open(mask_path).convert("L")
    binary = np.asarray(img) > 127
    if not binary.any():
        return 0.0
    return max(_skeleton_px(binary), 1.0) * edge / float(img.width)


def cell_measure(shape: str, rows: list[dict], edge: int) -> float:
    """The cell's median defect extent: skeleton length (px) for lines, mask area (px) otherwise."""
    if shape == "line":
        return statistics.median(skeleton_length(r["mask_path"], edge) for r in rows)
    return statistics.median(float(r["mask_coverage"]) * edge * edge for r in rows)


def sizes_for(shape: str, measure: float, rung: dict, edge: int = REFERENCE_EDGE) -> dict:
    """Region parameters of ``shape`` matching ``measure`` (see :func:`cell_measure`), others from ``rung``."""
    if shape == "blob":
        return {"radius": round(min(math.sqrt(measure / math.pi), MAX_BLOB_RADIUS_FRAC * edge), 1)}
    if shape == "line":
        return {
            "length": round(min(measure, MAX_LINE_LENGTH_FRAC * edge), 1),
            "width": float(rung["width"]),
            "branches": int(rung.get("branches", 0)),
        }
    if shape == "cluster":
        radius = float(rung["radius"])
        return {"count": max(1, int(round(measure / (math.pi * radius * radius)))), "spread": float(rung["spread"]), "radius": radius}
    raise ValueError(f"unknown shape {shape!r}")


def drawn_measure(block: dict, shape: str, stage: str, mode: str, edge: int, draws: int = 20) -> float:
    """What the current config draws for a cell, in :func:`cell_measure`'s unit (uniform lit crop)."""
    from PIL import Image

    from synevad.synthesis.regions import sample_region

    crop = np.full((edge, edge, 3), 60, dtype=np.uint8)
    spec = region_spec(block, shape=shape, stage=stage, mode=mode)
    values = []
    for seed in range(draws):
        binary = sample_region(crop, spec, seed=seed).binary
        if shape == "line":
            values.append(_skeleton_px(binary))
        else:
            values.append(float(binary.sum()))
    return float(np.median(values))


def _skeleton_px(binary: np.ndarray) -> float:
    from skimage.morphology import skeletonize

    skel = skeletonize(binary)
    straight = np.logical_and(skel[:, 1:], skel[:, :-1]).sum() + np.logical_and(skel[1:, :], skel[:-1, :]).sum()
    diagonal = np.logical_and(skel[1:, 1:], skel[:-1, :-1]).sum() + np.logical_and(skel[1:, :-1], skel[:-1, 1:]).sum()
    return float(straight + math.sqrt(2.0) * diagonal)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    block = cfg.get("negatives") or {}
    rows = read_rows(args.root)
    if not rows:
        raise SystemExit(f"no generation manifests under {args.root}/*/images")
    cells = usable_rows(rows, {s.strip() for s in args.statuses.split(",") if s.strip()})

    mode_sizes: dict[str, dict[str, dict]] = defaultdict(dict)
    print(f"{'class':<18} {'stage':<9} {'n':>4} {'defect':>13} {'region now':>11}  shape")
    for (cls, stage) in sorted(cells, key=lambda k: (k[0], STAGE_ORDER.index(k[1]) if k[1] in STAGE_ORDER else 99)):
        cell_rows = cells[(cls, stage)]
        shape = resolve_shape("", cls, block)
        now = drawn_measure(block, shape, stage, cls, args.edge)
        skipped = len(cell_rows) < args.min_samples
        measure = cell_measure(shape, cell_rows, args.edge)
        unit = "px long" if shape == "line" else "px2"
        flag = f"  (skipped: < {args.min_samples} masks)" if skipped else ""
        print(f"{cls:<18} {stage:<9} {len(cell_rows):>4} {measure:>6.0f} {unit:<7} {now:>6.0f} {unit:<7} {shape}{flag}")
        if not skipped:
            rung = dict(region_spec(block, shape=shape, stage=stage).sizes)
            mode_sizes[cls][stage] = sizes_for(shape, measure, rung, args.edge)

    if args.write_overrides:
        overlay = {
            cat.strip(): {"negatives": {"mode_sizes": {k: dict(v) for k, v in mode_sizes.items()}}}
            for cat in args.categories.split(",")
            if cat.strip()
        }
        Path(args.write_overrides).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_overrides).write_text(OmegaConf.to_yaml(overlay), encoding="utf-8")
        print(f"\nwrote {sum(len(v) for v in mode_sizes.values())} calibrated cell(s) to {args.write_overrides}")
    if not mode_sizes:
        print("\nno cell has enough usable masks; the config's sizes stay in force")
        return 0 if args.write_overrides else 1
    print("\n# paste under `negatives:` in the generation config")
    print(OmegaConf.to_yaml({"mode_sizes": {k: dict(v) for k, v in mode_sizes.items()}}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
