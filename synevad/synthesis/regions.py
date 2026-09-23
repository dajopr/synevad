"""Defect-shaped regions for negative-control inpainting (pure numpy/Pillow, no model).

A negative control has to differ from a synthetic defect in *content only*. The defects are
edited by FLUX and pasted back through an alpha around their estimated mask; a negative
inpaints a region with a clean, undamaged surface and pastes it back through the same
dilate/feather/colour/grain blend. If the region had the wrong shape or size, a detector could
still tell the two populations apart by *where* and *how much* was regenerated, so the region
mirrors the defect cell it stands in for:

* **shape** follows the defect type — ``blob`` (staining, erosion), ``line`` (a crack, with
  branches at higher severity), ``cluster`` (a pit field);
* **size** follows the severity stage, from :data:`DEFAULT_SIZES` (px at a 1024 px crop edge,
  scaled to the actual edge, jittered ±25% per region);
* **placement** is on the lit metal surface and away from the crop border, optionally inside a
  caller-supplied ROI mask — never in the black background or a shadowed opening, where a
  regenerated patch would be trivially invisible.

Everything is a deterministic function of the seed, so a generation worker and a later blend
of the same job always agree on the region even without reading it back from disk.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SHAPES = ("blob", "line", "cluster")

# Sizes at a 1024 px crop edge. `radius` / `length` / `width` / `spread` are px, `count` is
# the number of pits, `branches` the number of side cracks off the main line.
DEFAULT_SIZES: dict[str, dict[str, dict[str, float]]] = {
    "blob": {
        "minimal": {"radius": 24},
        "slight": {"radius": 42},
        "moderate": {"radius": 75},
        "severe": {"radius": 120},
    },
    "line": {
        "minimal": {"length": 90, "width": 2.0, "branches": 0},
        "slight": {"length": 170, "width": 2.5, "branches": 0},
        "moderate": {"length": 300, "width": 3.5, "branches": 1},
        "severe": {"length": 480, "width": 5.0, "branches": 2},
    },
    "cluster": {
        "minimal": {"count": 4, "spread": 14, "radius": 3.0},
        "slight": {"count": 8, "spread": 24, "radius": 4.0},
        "moderate": {"count": 16, "spread": 40, "radius": 5.0},
        "severe": {"count": 30, "spread": 65, "radius": 6.5},
    },
}
REFERENCE_EDGE = 1024
_JITTER = 0.25
_SUPERSAMPLE = 4


@dataclass(frozen=True)
class RegionSpec:
    """How to draw and place one region."""

    shape: str
    stage: str
    sizes: Mapping[str, float]
    surface_level: float = 0.25
    margin_px: int = 96
    min_inside: float = 0.9
    max_tries: int = 50


@dataclass(frozen=True)
class Region:
    """A sampled region: the binary mask plus how it was drawn."""

    binary: np.ndarray  # bool (H, W)
    centre: tuple[int, int]
    inside_fraction: float  # share of the region on the placement surface
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.binary.mean())

    def image(self) -> Image.Image:
        return Image.fromarray(self.binary.astype(np.uint8) * 255, mode="L")


# --- config -----------------------------------------------------------------


def resolve_shape(prompt_shape: str | None, mode: str, block: Mapping | None) -> str:
    """The region shape for a defect mode.

    Precedence: the prompt entry's own ``region_shape``, then ``negatives.shapes[mode]``, then
    ``negatives.default_shape``, then ``blob``. An unknown name fails here rather than drawing
    something the config did not ask for.
    """
    block = block or {}
    shape = (
        (prompt_shape or "").strip()
        or str((block.get("shapes") or {}).get(mode) or "").strip()
        or str(block.get("default_shape") or "").strip()
        or "blob"
    ).lower()
    if shape not in SHAPES:
        raise ValueError(f"region shape must be one of {', '.join(SHAPES)}, got {shape!r}")
    return shape


def region_spec(
    block: Mapping | None, *, shape: str, stage: str, mode: str | None = None
) -> RegionSpec:
    """A :class:`RegionSpec` from a config ``negatives:`` block.

    Sizes resolve, most specific first: ``mode_sizes[mode][stage]`` (a defect type whose extent
    differs from its shape's default, e.g. a single stone strike drawn as a small blob), then
    ``sizes[shape][stage]``, then :data:`DEFAULT_SIZES`. Each level is a partial merge.
    """
    block = block or {}
    sizes = copy.deepcopy(DEFAULT_SIZES)
    for shp, stages in (block.get("sizes") or {}).items():
        for stg, values in (stages or {}).items():
            sizes.setdefault(str(shp), {}).setdefault(str(stg), {}).update(dict(values or {}))
    by_stage = sizes.get(shape) or {}
    per_mode = ((block.get("mode_sizes") or {}).get(mode) or {}) if mode else {}
    if stage not in by_stage and stage not in per_mode:
        raise ValueError(
            f"no region size for shape {shape!r} at stage {stage!r} "
            f"(known stages: {', '.join(by_stage) or 'none'})"
        )
    merged = {**by_stage.get(stage, {}), **dict(per_mode.get(stage) or {})}
    placement = block.get("placement") or {}
    return RegionSpec(
        shape=shape,
        stage=stage,
        sizes=merged,
        surface_level=float(placement.get("surface_level", 0.25)),
        margin_px=int(placement.get("margin_px", 96)),
        min_inside=float(placement.get("min_inside", 0.9)),
        max_tries=int(placement.get("max_tries", 50)),
    )


# --- placement ----------------------------------------------------------------


def placement_mask(
    clean: np.ndarray, spec: RegionSpec, roi: np.ndarray | None = None
) -> np.ndarray:
    """Where a region may sit: lit surface, away from the border, inside ``roi`` if given.

    "Lit" is smoothed luminance above ``surface_level`` of the way from the dark floor (5th
    percentile) to the bright level (95th percentile) of the pixels the margin and ROI allow.
    Relative to the crop's own range, so it works on the very dark raw frames as on a brightly
    lit object — and unlike a plain quantile it does not fall onto the dark plateau when most
    of the crop is background.
    """
    arr = np.asarray(clean, dtype=np.float64)
    h, w = arr.shape[:2]
    lum = arr.mean(axis=2) if arr.ndim == 3 else arr
    # The smoothing sigma is ~1/40 of the edge, so the map carries no detail a 4x-reduced
    # computation would lose; it keeps placement cheap enough to run once per job.
    f = max(1, min(h, w) // 256)
    small = lum[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    small = ndimage.gaussian_filter(small, max(1.0, min(h, w) / 40.0 / f))
    lum = np.kron(small, np.ones((f, f)))
    lum = np.pad(lum, ((0, h - lum.shape[0]), (0, w - lum.shape[1])), mode="edge")
    scale = min(h, w) / REFERENCE_EDGE
    margin = int(round(spec.margin_px * scale))
    allowed = np.zeros((h, w), dtype=bool)
    allowed[margin : h - margin, margin : w - margin] = True
    if roi is not None:
        allowed &= np.asarray(roi, dtype=bool)
    if not allowed.any():
        raise ValueError("no placement area left after the border margin and ROI")
    lo, hi = np.quantile(lum[allowed], [0.05, 0.95])
    cut = lo + spec.surface_level * (hi - lo)
    surface = allowed & (lum > cut)
    return surface if surface.any() else allowed


# --- shapes -------------------------------------------------------------------


def _jitter(value: float, rng: np.random.Generator) -> float:
    return float(value) * float(rng.uniform(1.0 - _JITTER, 1.0 + _JITTER))


def _rasterise(
    size: tuple[int, int],
    lines: list[tuple[list[tuple[float, float]], float]],
    discs: list[tuple[float, float, float]],
) -> np.ndarray:
    """Anti-aliased polylines ``(points, width)`` and discs ``(x, y, r)`` as a bool mask.

    Drawn supersampled on a canvas covering only the primitives' bounding box (not the whole
    crop), downsampled with a box filter and cut at half coverage — so a thin crack stays
    connected but does not bloat, and the cost scales with the region, not the crop.
    """
    w, h = size
    pad = 2.0
    xs, ys = [], []
    for pts, width in lines:
        xs += [p[0] - width - pad for p in pts] + [p[0] + width + pad for p in pts]
        ys += [p[1] - width - pad for p in pts] + [p[1] + width + pad for p in pts]
    for x, y, r in discs:
        xs += [x - r - pad, x + r + pad]
        ys += [y - r - pad, y + r + pad]
    out = np.zeros((h, w), dtype=bool)
    if not xs:
        return out
    x0, x1 = max(0, int(np.floor(min(xs)))), min(w, int(np.ceil(max(xs))))
    y0, y1 = max(0, int(np.floor(min(ys)))), min(h, int(np.ceil(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return out
    up = _SUPERSAMPLE
    canvas = Image.new("L", ((x1 - x0) * up, (y1 - y0) * up), 0)
    draw = ImageDraw.Draw(canvas)
    for pts, width in lines:
        draw.line(
            [((x - x0) * up, (y - y0) * up) for x, y in pts],
            fill=255,
            width=max(1, int(round(width * up))),
            joint="curve",
        )
    for x, y, r in discs:
        draw.ellipse([(x - x0 - r) * up, (y - y0 - r) * up, (x - x0 + r) * up, (y - y0 + r) * up], fill=255)
    window = canvas.resize((x1 - x0, y1 - y0), Image.Resampling.BOX)
    out[y0:y1, x0:x1] = np.asarray(window, dtype=np.float64) / 255.0 >= 0.5
    return out


def _walk(x: float, y: float, angle: float, length: float, step: float, rng) -> list[tuple[float, float]]:
    pts = [(x, y)]
    for _ in range(max(1, int(length / step))):
        angle += float(rng.normal(0.0, 0.2))
        x, y = x + step * np.cos(angle), y + step * np.sin(angle)
        pts.append((x, y))
    return pts


def draw_shape(
    size: tuple[int, int], centre: tuple[int, int], spec: RegionSpec, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterise one region of ``spec.shape`` centred on ``centre``; returns ``(bool mask, params)``."""
    w, h = size
    cx, cy = centre
    scale = min(w, h) / REFERENCE_EDGE
    s = spec.sizes

    if spec.shape == "blob":
        radius = max(2.0, _jitter(s["radius"], rng) * scale)
        pad = int(np.ceil(radius * 1.6)) + 2
        x0, x1 = max(0, cx - pad), min(w, cx + pad)
        y0, y1 = max(0, cy - pad), min(h, cy + pad)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        noise = ndimage.gaussian_filter(rng.standard_normal(yy.shape), max(1.0, radius / 4.0))
        noise /= noise.std() + 1e-9
        field_ = 1.0 - np.hypot(xx - cx, yy - cy) / radius + 0.3 * noise
        out = np.zeros((h, w), dtype=bool)
        out[y0:y1, x0:x1] = ndimage.binary_fill_holes(field_ > 0.0)
        labels, n = ndimage.label(out)
        if n > 1:  # keep the component under the centre (or the largest)
            keep = labels[cy, cx] or int(np.argmax(np.bincount(labels.ravel())[1:]) + 1)
            out = labels == keep
        return out, {"radius": round(radius, 2)}

    if spec.shape == "line":
        length = max(8.0, _jitter(s["length"], rng) * scale)
        width = max(1.0, _jitter(s["width"], rng) * scale)
        step = max(1.0, 4.0 * scale)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        forward = _walk(cx, cy, angle, length / 2.0, step, rng)
        backward = _walk(cx, cy, angle + np.pi, length / 2.0, step, rng)
        main = backward[::-1] + forward[1:]
        lines = [(main, width)]
        n_branches = int(s.get("branches", 0))
        for _ in range(n_branches):
            bx, by = main[int(rng.integers(len(main) // 4, max(len(main) // 4 + 1, 3 * len(main) // 4)))]
            b_angle = angle + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.5, 1.1))
            branch = _walk(bx, by, b_angle, length * float(rng.uniform(0.25, 0.5)), step, rng)
            lines.append((branch, 0.7 * width))
        return _rasterise((w, h), lines, []), {
            "length": round(length, 1), "width": round(width, 2), "branches": n_branches,
        }

    if spec.shape == "cluster":
        count = max(1, int(round(_jitter(s["count"], rng))))
        spread = max(1.0, _jitter(s["spread"], rng) * scale)
        base_r = max(1.0, float(s["radius"]) * scale)
        discs = [
            (cx + rng.normal(0.0, spread), cy + rng.normal(0.0, spread), base_r * float(rng.uniform(0.7, 1.3)))
            for _ in range(count)
        ]
        return _rasterise((w, h), [], discs), {
            "count": count, "spread": round(spread, 1), "radius": round(base_r, 2),
        }

    raise ValueError(f"unknown region shape {spec.shape!r}")


def sample_region(
    clean: Image.Image | np.ndarray,
    spec: RegionSpec,
    seed: int,
    roi: np.ndarray | None = None,
) -> Region:
    """Draw a region for ``clean`` deterministically from ``seed``.

    Retries placement up to ``spec.max_tries`` times until at least ``spec.min_inside`` of the
    region lies on the placement surface; otherwise keeps the best attempt, recording its
    ``inside_fraction`` so a poor placement is visible in the manifest rather than hidden.
    """
    arr = np.asarray(clean.convert("RGB") if isinstance(clean, Image.Image) else clean)
    h, w = arr.shape[:2]
    surface = placement_mask(arr, spec, roi)
    ys, xs = np.nonzero(surface)
    rng = np.random.default_rng(int(seed))
    best: Region | None = None
    for attempt in range(max(1, spec.max_tries)):
        k = int(rng.integers(len(ys)))
        centre = (int(xs[k]), int(ys[k]))
        binary, params = draw_shape((w, h), centre, spec, rng)
        if not binary.any():
            continue
        inside = float(np.logical_and(binary, surface).sum()) / float(binary.sum())
        region = Region(
            binary=binary,
            centre=centre,
            inside_fraction=round(inside, 4),
            params={**params, "shape": spec.shape, "stage": spec.stage, "attempt": attempt},
        )
        if best is None or inside > best.inside_fraction:
            best = region
        if inside >= spec.min_inside:
            return region
    if best is None:
        raise ValueError(f"could not draw a non-empty {spec.shape} region")
    return best


def inpaint_support(binary: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    """The area the inpainter must regenerate so the composite alpha only covers new pixels.

    The paste alpha is ``binary`` dilated by ``dilate_px + feather_px`` and then Gaussian
    feathered with ``sigma = feather_px`` (:func:`synevad.synthesis.masks._dilate_feather`), which scipy
    truncates at 4 sigma; growing by that further ``4 * feather_px`` covers the whole tail, so
    no composited pixel — however small its weight — comes from outside what was repainted.
    """
    radius = int(dilate_px) + 5 * int(feather_px)
    b = np.asarray(binary, dtype=bool)
    if radius <= 0:
        return b
    return ndimage.maximum_filter(b.astype(np.uint8), size=2 * radius + 1) > 0
