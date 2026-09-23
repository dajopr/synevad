"""Geometric mask refinement for the defect-mask pipeline.

Step 3 of :func:`synevad.synthesis.masks.estimate_mask`. The refined mask is always band-constrained
around the seed: ``M = M_ref ∩ dilate(M_seed, r)``. The refiner may drop halo pixels inside
the seed; it can never jump outside the band, so refinement cannot relocate a defect.

The one refiner is the **guided filter** (:func:`guided_refine`), which snaps the coarse
ResNet-18 seed onto the photometric residual's edges so pixel ground truth is not stuck on
the stride-8 feature grid. It is applied by the estimator itself; this module owns the
filter and the shared geometry helpers.

Pure numpy/scipy/Pillow — no model, no torch.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks; 0 when both are empty."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return (inter / union) if union else 0.0


def _match_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask onto ``shape`` (H, W)."""
    arr = np.asarray(mask, dtype=bool)
    if arr.shape == shape:
        return arr
    return (
        np.asarray(
            Image.fromarray(arr.astype(np.uint8) * 255, mode="L").resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
        )
        > 0
    )


def _dilate_band(seed: np.ndarray, radius: int) -> np.ndarray:
    seed_b = np.asarray(seed, dtype=bool)
    if radius > 0:
        return ndimage.binary_dilation(seed_b, iterations=int(radius))
    return seed_b


def intersect_constrain(
    seed: np.ndarray, refined: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``M = refined ∩ dilate(seed, r)``.

    The refiner may drop seed pixels (shrink a halo); it cannot jump outside the band.
    Returns ``(constrained, clipped)`` — the two are identical under intersect.
    """
    seed_b = np.asarray(seed, dtype=bool)
    ref_b = _match_shape(refined, seed_b.shape)
    band = _dilate_band(seed_b, radius)
    clipped = np.logical_and(ref_b, band)
    return clipped, clipped


def guided_refine(
    score: np.ndarray, guide: np.ndarray, *, radius: int = 8, eps: float = 1e-4
) -> np.ndarray:
    """Transfer ``score`` onto ``guide``'s edges via the guided filter (He et al. 2010).

    Snaps a coarse feature-map seed onto residual-magnitude edges. The filter is O(n)
    box-filter arithmetic, not a sliding window, so it is cheap at 1024². If ``score`` is a
    different shape it is bilinearly resampled to the guide first.
    """
    if score.shape != guide.shape:
        score = np.asarray(
            Image.fromarray(score.astype(np.float32), mode="F").resize(
                (guide.shape[1], guide.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
    g = guide.astype(np.float64)
    span = float(g.max() - g.min())
    g = (g - g.min()) / (span + 1e-8)
    p = score.astype(np.float64)
    size = 2 * int(radius) + 1

    def mean(arr: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(arr, size=size, mode="nearest")

    mean_g, mean_p = mean(g), mean(p)
    var_g = mean(g * g) - mean_g * mean_g
    cov_gp = mean(g * p) - mean_g * mean_p
    a = cov_gp / (var_g + float(eps))
    b = mean_p - a * mean_g
    return mean(a) * g + mean(b)
