"""Cropping helpers (pure Pillow, no model).

A "crop" is the box of the source frame that the editor and the mask estimator both
see; :func:`crop_image` turns that box into the image FLUX is handed, and
:class:`CropPlan` turns a config ``cropping:`` block into the boxes for one frame.

Whole-frame (``crop_size: null``) stays the default because that is what every MVTec AD
run did: those frames are already square at 900-1024 px, so "crop" was the full-frame box
and ``resize_to`` a lossless-ish square resample. MVTec AD 2 frames are 2448x2048 through
4224x1056, where the same path both squashes the aspect ratio and downsamples 2.4-4x --
far enough that the hairline defects the taxonomy asks for land below the editor's
resolution. Those categories crop instead, and ``resize_to`` may be a ``[w, h]`` pair for
the ones that have to stay whole-frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

_RESAMPLE = Image.Resampling.LANCZOS

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), PIL `.crop` convention
Size = tuple[int, int]  # (W, H), PIL `.size` convention

LAYOUTS = ("full", "center", "grid", "centers")

# A tiling axis whose leftover is under this fraction of the window is one centred
# window, not two: a 1056 px axis windowed at 1024 is a single crop with a 32 px
# residue, and splitting it would generate two near-identical frames.
_MIN_STEP_FRAC = 0.10

# FLUX.2 packs latents on a 16 px grid; an edge that is not a multiple of 16 fails
# deep inside the pipeline, so the config parser rejects it here instead.
_EDGE_MULTIPLE = 16


# --- boxes ------------------------------------------------------------------


def full_box(img_size: Size) -> Box:
    """The whole-frame box for an image of ``img_size``."""
    w, h = img_size
    return (0, 0, int(w), int(h))


def parse_centers(items: Any) -> tuple[tuple[int, int], ...]:
    """Parse crop centres given as ``"CX,CY"`` strings or ``[cx, cy]`` pairs."""
    if items is None:
        return ()
    if isinstance(items, (str, bytes)):
        items = [items]
    out: list[tuple[int, int]] = []
    for it in items:
        if isinstance(it, (str, bytes)):
            parts = str(it).split(",")
        elif isinstance(it, Sequence):
            parts = list(it)
        else:
            raise SystemExit(f"crop centre must be 'CX,CY' or [cx, cy], got: {it!r}")
        if len(parts) != 2:
            raise SystemExit(f"crop centre must have two values, got: {it!r}")
        try:
            cx, cy = (int(v) for v in parts)
        except (TypeError, ValueError):
            raise SystemExit(f"crop centre must be integers, got: {it!r}") from None
        out.append((cx, cy))
    return tuple(out)


def boxes_from_centers(
    centers: Sequence[tuple[int, int]], crop_size: int, img_size: Size
) -> list[Box]:
    """Fixed square crop boxes centred at ``centers``, clamped to stay inside the image.

    A box is ``crop_size`` square; if ``crop_size`` exceeds an image dimension the box
    spans the whole image on that axis. Deterministic -- the same centres yield the same
    boxes on every run.
    """
    w, h = img_size
    half = int(crop_size) // 2
    boxes: list[Box] = []
    for cx, cy in centers:
        if crop_size >= w:
            x0, x1 = 0, w
        else:
            x0 = min(max(0, cx - half), w - crop_size)
            x1 = x0 + crop_size
        if crop_size >= h:
            y0, y1 = 0, h
        else:
            y0 = min(max(0, cy - half), h - crop_size)
            y1 = y0 + crop_size
        boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes


def center_box(img_size: Size, crop_size: int) -> Box:
    """The single ``crop_size`` square box at the centre of the frame."""
    w, h = img_size
    return boxes_from_centers([(w // 2, h // 2)], crop_size, img_size)[0]


def _axis_starts(length: int, size: int, stride: int) -> list[int]:
    """Window starts covering ``length`` with windows of ``size``, evenly spread.

    The starts always begin at 0 and end at ``length - size``, so the union covers the
    axis end to end; ``stride`` only sets how many windows that takes. An axis whose
    leftover is a sliver (under :data:`_MIN_STEP_FRAC` of the window) collapses to one
    centred window rather than two that overlap by all but a few pixels.
    """
    if size >= length:
        return [0]
    span = length - size
    if span <= max(1, int(round(size * _MIN_STEP_FRAC))):
        return [span // 2]
    n = math.ceil(span / max(1, stride)) + 1
    return [round(i * span / (n - 1)) for i in range(n)]


def grid_boxes(img_size: Size, crop_size: int, *, overlap: float = 0.0) -> list[Box]:
    """Square boxes tiling the frame, row-major, with a fractional ``overlap``.

    ``crop_size`` is clamped to the short edge so every box stays square. Boxes cover the
    frame end to end; ``overlap`` (0-0.9) shrinks the stride and so adds boxes.
    """
    w, h = img_size
    if not 0.0 <= overlap < 1.0:
        raise SystemExit(f"cropping.overlap must be in [0, 1), got {overlap}")
    size = min(int(crop_size), int(w), int(h))
    stride = max(1, int(round(size * (1.0 - overlap))))
    return [
        (x0, y0, x0 + size, y0 + size)
        for y0 in _axis_starts(int(h), size, stride)
        for x0 in _axis_starts(int(w), size, stride)
    ]


# --- plan -------------------------------------------------------------------


@dataclass(frozen=True)
class CropPlan:
    """How one source frame is turned into the boxes a run edits.

    The default is the whole frame as a single box, which is what the MVTec AD configs
    have always done. ``layout`` is only consulted once ``crop_size`` is set.
    """

    crop_size: int | None = None
    layout: str = "full"
    centers: tuple[tuple[int, int], ...] = ()
    overlap: float = 0.0

    def boxes(self, img_size: Size) -> list[Box]:
        """The crop boxes for a frame of ``img_size``, in a stable order."""
        if self.layout == "full" or self.crop_size is None:
            return [full_box(img_size)]
        if self.layout == "centers":
            return boxes_from_centers(self.centers, self.crop_size, img_size)
        if self.layout == "grid":
            return grid_boxes(img_size, self.crop_size, overlap=self.overlap)
        return [center_box(img_size, self.crop_size)]

    def describe(self) -> str:
        """One-line summary for the run's plan log."""
        if self.layout == "full" or self.crop_size is None:
            return "whole frame"
        if self.layout == "centers":
            return f"{len(self.centers)} centred {self.crop_size}px crop(s)"
        if self.layout == "grid":
            return f"{self.crop_size}px grid (overlap {self.overlap:g})"
        return f"centre {self.crop_size}px crop"


def crop_plan(block: Mapping[str, Any] | None) -> CropPlan:
    """Build a :class:`CropPlan` from a config ``cropping:`` block.

    ``crop_centers`` implies ``layout: centers``; a null ``crop_size`` implies
    ``layout: full`` whatever else is set, so the historical
    ``{crop_size: 1024, crop_centers: null}`` block keeps meaning "whole frame" only
    when ``layout`` is left at its default.
    """
    block = dict(block or {})
    raw_size = block.get("crop_size")
    crop_size = None if raw_size is None else int(raw_size)
    if crop_size is not None and crop_size < 1:
        raise SystemExit(f"cropping.crop_size must be >= 1, got {crop_size}")
    centers = parse_centers(block.get("crop_centers"))
    layout = str(block.get("layout") or ("centers" if centers else "full")).lower()
    if layout not in LAYOUTS:
        raise SystemExit(
            f"cropping.layout must be one of {', '.join(LAYOUTS)}, got {layout!r}"
        )
    if layout == "centers" and not centers:
        raise SystemExit("cropping.layout: centers needs cropping.crop_centers")
    if layout != "full" and crop_size is None:
        raise SystemExit(f"cropping.layout: {layout} needs cropping.crop_size")
    return CropPlan(
        crop_size=crop_size,
        layout=layout,
        centers=centers,
        overlap=float(block.get("overlap") or 0.0),
    )


# --- resize -----------------------------------------------------------------


def resize_size(resize_to: int | Sequence[int] | None) -> Size | None:
    """Normalise ``resize_to`` to ``(w, h)``; a scalar means a square."""
    if resize_to is None:
        return None
    if isinstance(resize_to, (int, float)):
        side = int(resize_to)
        return (side, side)
    values = [int(v) for v in resize_to]
    if len(values) != 2:
        raise SystemExit(f"resize_to must be a scalar or [w, h], got {resize_to!r}")
    return (values[0], values[1])


def parse_resize_to(value: Any) -> Size:
    """Config-level ``resize_to``: a square side or ``[w, h]``, both 16-aligned."""
    size = resize_size(value)
    if size is None:
        raise SystemExit("cropping.resize_to is required")
    w, h = size
    if w < 1 or h < 1:
        raise SystemExit(f"cropping.resize_to must be positive, got {value!r}")
    if w % _EDGE_MULTIPLE or h % _EDGE_MULTIPLE:
        raise SystemExit(
            f"cropping.resize_to must be a multiple of {_EDGE_MULTIPLE} on both edges "
            f"(FLUX.2 packs latents on that grid), got {w}x{h}"
        )
    return (w, h)


def aspect_warning(
    img_size: Size, box: Box, resize_to: int | Sequence[int] | None
) -> str | None:
    """Warn when a box is about to be resampled to a materially different aspect.

    Squashing is silent and looks like a slightly odd render rather than a bug, so the
    run says it out loud. 2% tolerance absorbs the rounding of an odd edge.
    """
    target = resize_size(resize_to)
    if target is None:
        return None
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    src = bw / bh
    dst = target[0] / target[1]
    if abs(src - dst) <= 0.02 * max(src, dst):
        return None
    return (
        f"crop {bw}x{bh} (aspect {src:.3f}) resampled to {target[0]}x{target[1]} "
        f"(aspect {dst:.3f}) — the edit is stretched; set cropping.crop_size for a "
        f"square crop, or cropping.resize_to: [w, h] to keep the frame's aspect "
        f"(source {img_size[0]}x{img_size[1]})"
    )


def crop_image(
    img: Image.Image, box: Box, resize_to: int | Sequence[int] | None = None
) -> Image.Image:
    """Crop ``img`` to ``box`` and optionally resample it to ``resize_to``.

    ``resize_to`` is a square side or a ``(w, h)`` pair.
    """
    crop = img.crop(box)
    target = resize_size(resize_to)
    if target is not None:
        crop = crop.resize(target, _RESAMPLE)
    return crop
