"""Defect-mask pipeline: dense change-map seed, constrained guided refine.

The mask is the pipeline's *pixel ground truth*. Stages:

0. **Change gate.** ``semantic_fn`` supplies a dense backbone feature-map change
   map (DINOv3 patch tokens or ImageNet ResNet-18) over the ``(C, E)`` pair, smoothed by
   ``smooth_px`` and thresholded at
   ``background + k·sigma`` (absolute, not quantile), floored at ``min_change``. No
   surviving component → ``no_defect``.
1. **Proposals.** Connected components of the thresholded map, filtered by ``min_area_px``.
2. **Guided refine** (optional, ``guide_refine``). The feature map decides *support*; the
   photometric residual places the *edge*. :func:`synevad.synthesis.refine.guided_refine` snaps the
   coarse seed onto residual edges, and the result is intersect-constrained to
   ``dilate(seed, dilate_r)`` so refinement can drop the halo but never jump outside the
   band. A rejected refinement falls back to the seed.
3. **Acceptance.** Area fraction vs the stage's expected-extent range.

Every stage appends a record to ``MaskResult.trace`` (and the manifest's ``mask_trace``):
what it did (``seeded`` / ``shrunk`` / ``fallback`` / …), coverage and component count
after, plus step-specific extras.

**A noise floor that survives large defects.** The background level in stage 0 is *not* the
plain median over the region — that assumes the defect covers less than half of it, and past
that point the estimator does not degrade, it *inverts*: the median moves inside the change,
the threshold rises above the defect, and the candidate comes back with an empty mask and a
confident ``no_defect``. On the v3 corpus that silently cost 812 of 5360 candidates their
pixel ground truth, concentrated exactly where the benchmark is hardest — tile kept 6 of 100
``severe``, leather 22 of 100. :func:`_noise_level` instead seeds the background at a low
quantile and refines it by clipping, which moves the breakdown point to a configurable
contamination fraction.

That is not an optional refinement of stage 3 but a precondition for it: the ``object_scale``
bucket accepts masks up to 0.80 coverage, and a global median+MAD cannot *detect* anything
past ~0.50. Under it, every ``severe`` candidate between those two numbers returns empty and
is tagged ``no_defect``, so two thirds of the bucket's range is unreachable.

A non-``ok`` status is a statement about *this candidate*, not an error: the caller
persists it so the run keeps its full audit trail. Empty masks composite to the clean crop
bit-identically.

The ``valid`` argument supports re-masking an already-composited pair (see
re-estimating from a blended crop): outside the original compositing alpha it is
bit-identical to the clean crop, so the change there is exactly zero and would drag the
noise estimate to zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from synevad.synthesis.crops import Box
from synevad.synthesis.refine import binary_iou, guided_refine, intersect_constrain

_RESAMPLE = Image.Resampling.LANCZOS

# Status values on MaskResult. Only OK carries valid pixel ground truth.
STATUS_OK = "ok"
STATUS_NO_DEFECT = "no_defect"  # nothing cleared the noise-calibrated threshold
STATUS_OVER_COVERAGE = "over_coverage"  # change too diffuse/large for the extent range
STATUS_UNDER_COVERAGE = "under_coverage"  # localised, but too small for the stage claimed

# The stage that produced the geometry, recorded on MaskResult.mask_stage.
SEED_STAGE = "resnet18"
POST_BLEND_STAGE = "resnet18_post"
SEED_STAGES = frozenset({"resnet18", "dinov3"})
GUIDED_STAGES = frozenset({"guided", "guided_fallback"})
STAGE_TO_EXTENT = {
    "minimal": "localized",
    "slight": "localized",
    "moderate": "regional",
    "severe": "object_scale",
}
DEFAULT_EXTENT_RANGES: dict[str, tuple[float, float]] = {
    "localized": (0.0002, 0.05),
    "regional": (0.02, 0.25),
    "object_scale": (0.10, 0.80),
}
# Residual-threshold grid for inclusion-frequency uncertainty. Always unioned with the
# operating ``k_sigma`` so the seed mask is one of the votes.
DEFAULT_K_SWEEP: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
_FREQ_ONE = 1.0 - 1e-12

SemanticFn = Callable[[Image.Image, Image.Image], np.ndarray]


@dataclass(frozen=True)
class MaskResult:
    """One candidate's estimated defect mask plus the evidence for trusting it.

    ``binary`` is the ground-truth mask (operating ``k_sigma``, after refine). ``soft``
    is threshold inclusion frequency in ``[0, 1]`` (QC only — not compositing).
    ``alpha`` is the (dilated, feathered) compositing weight, larger than ``binary`` by
    construction. ``coverage`` is always the mask area fraction (logged for every
    candidate, including rejects). ``mask_stage`` records which stage produced the final
    geometry. ``trace`` is the per-step audit (seed → refine → accept, plus post-blend
    when that pass ran). ``mask_uncertainty`` is ``area(flicker) / area(core)``;
    ``None`` when the core is empty.
    """

    soft: Image.Image  # L, inclusion frequency 0..255 over the k-sweep
    binary: Image.Image  # L, 0/255 ground-truth mask
    alpha: Image.Image  # L, compositing weight (binary dilated + feathered)
    coverage: float  # fraction of the crop inside `binary` — always set
    components: int  # surviving connected components in `binary`
    status: str  # OK | NO_DEFECT | OVER_COVERAGE | UNDER_COVERAGE
    mask_stage: str = SEED_STAGE  # dinov3 | resnet18 | guided | guided_fallback | *_post
    expected_extent: str | None = None
    extent_lo: float | None = None
    extent_hi: float | None = None
    mask_uncertainty: float | None = None  # area(flicker) / area(core)
    mask_core_coverage: float = 0.0
    mask_flicker_coverage: float = 0.0
    # First-pass compositing mask, filled when a post-blend pass replaced
    # ``binary`` / ``status`` with a measurement of the stored composite.
    paste_status: str | None = None
    paste_coverage: float | None = None
    paste_stage: str | None = None
    paste_components: int | None = None
    paste_binary: Image.Image | None = None
    # Per-step audit of the estimator, one record per stage that ran.
    # See :func:`_mask_step`. Empty only on hand-built results that never ran the pipeline.
    trace: tuple[dict, ...] = ()

    @property
    def ok(self) -> bool:
        """True when this mask is usable as pixel ground truth."""
        return self.status == STATUS_OK

    def manifest_fields(self) -> dict:
        """QC columns the generation/remask manifests record for every candidate.

        ``mask_trace`` is the per-step audit (see :func:`_mask_step`). Existing
        scalar columns stay the source of truth for filters; the trace is why they
        landed that way.
        """
        pasted = self.paste_coverage if self.paste_coverage is not None else self.coverage
        fields = {
            "mask_status": self.status,
            "mask_coverage": round(self.coverage, 6),
            "mask_components": self.components,
            "mask_stage": self.mask_stage,
            "used_mask_composite": pasted > 0.0,
            "expected_extent": self.expected_extent,
            "extent_lo": self.extent_lo,
            "extent_hi": self.extent_hi,
            "mask_uncertainty": (
                None if self.mask_uncertainty is None else round(self.mask_uncertainty, 6)
            ),
            "mask_core_coverage": round(self.mask_core_coverage, 6),
            "mask_flicker_coverage": round(self.mask_flicker_coverage, 6),
            "mask_trace": [dict(s) for s in self.trace],
        }
        if self.paste_coverage is not None:
            fields["paste_status"] = self.paste_status
            fields["paste_coverage"] = round(self.paste_coverage, 6)
            fields["paste_stage"] = self.paste_stage
            fields["paste_components"] = self.paste_components
        return fields


def with_post_blend(
    paste: MaskResult, measured: MaskResult, *, post_stage: str | None = None
) -> MaskResult:
    """Keep the compositing ``alpha`` from ``paste``; pixel GT comes from ``measured``.

    The first pass decides where to paste. After that paste, ``measured`` is a fresh
    estimate on ``(clean, blended)`` — what actually changed in the stored crop.
    ``used_mask_composite`` still follows the paste, so an empty post-blend mask
    does not pretend the composite was a full-crop fallback.
    """
    if post_stage is None:
        post_stage = POST_BLEND_STAGE
    stage = (
        post_stage
        if measured.mask_stage in SEED_STAGES | GUIDED_STAGES
        else measured.mask_stage
    )
    return replace(
        measured,
        alpha=paste.alpha,
        mask_stage=stage,
        paste_status=paste.status,
        paste_coverage=paste.coverage,
        paste_stage=paste.mask_stage,
        paste_components=paste.components,
        paste_binary=paste.binary,
        trace=tuple(_tag_pass(paste.trace, "paste") + _tag_pass(measured.trace, "post_blend")),
    )


def _tag_pass(steps: Sequence[Mapping], pass_name: str) -> list[dict]:
    """Copy ``steps`` with ``pass`` set so a two-pass trace is not two unlabeled seeds."""
    return [{**dict(s), "pass": pass_name} for s in steps]


def extent_for_stage(stage: str | None) -> str:
    """Map a severity stage name to an expected-extent bucket; unknown → ``localized``."""
    if not stage:
        return "localized"
    return STAGE_TO_EXTENT.get(str(stage).strip().lower(), "localized")


def _poly_basis(shape: tuple[int, int], degree: int) -> np.ndarray:
    """Design matrix (N, K) of 2-D monomials up to ``degree`` on normalised [-1,1] coordinates."""
    h, w = shape
    y = (np.arange(h, dtype=np.float64) / max(h - 1, 1)) * 2.0 - 1.0
    x = (np.arange(w, dtype=np.float64) / max(w - 1, 1)) * 2.0 - 1.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    terms = [
        (xx**i) * (yy**j) for i in range(degree + 1) for j in range(degree + 1 - i)
    ]
    return np.stack([t.ravel() for t in terms], axis=1)


def _detrend(
    diff: np.ndarray, degree: int, trim: float, valid: np.ndarray | None, fit_stride: int = 4
) -> np.ndarray:
    """Remove the editor's global photometric drift from a residual, per channel.

    The drift to remove is a global exposure / white-balance / vignetting shift, which a
    low-order polynomial surface models directly. The obvious alternative — subtracting a
    heavily blurred copy of the residual — *rings*: around a compact defect the blur places a
    wide, low-amplitude bump of opposite sign, and the magnitude of that bump clears the
    detection threshold, wrapping every defect in a halo of false positives (~15% of the frame
    for a small pit, in the unit tests). A polynomial cannot produce that halo, and it cannot
    absorb a localised blotch either, so diffuse defects survive detrending instead of being
    silently removed along with the drift.

    ``trim`` refits after dropping the most deviant fraction of pixels, so a large defect does
    not tilt the surface toward itself. The fit is subsampled by ``fit_stride`` — estimating a
    handful of coefficients does not need every pixel.
    """
    h, w, channels = diff.shape
    basis = _poly_basis((h, w), degree)
    flat_valid = None if valid is None else valid.ravel()

    sub = np.zeros(h * w, dtype=bool)
    sub.reshape(h, w)[::fit_stride, ::fit_stride] = True
    if flat_valid is not None:
        sub &= flat_valid
    if sub.sum() <= basis.shape[1]:  # too few points to fit: fall back to every valid pixel
        sub = np.ones(h * w, dtype=bool) if flat_valid is None else flat_valid.copy()

    out = np.empty_like(diff)
    for c in range(channels):
        values = diff[..., c].ravel()
        coef, *_ = np.linalg.lstsq(basis[sub], values[sub], rcond=None)
        if trim > 0:
            deviation = np.abs(values - basis @ coef)
            keep = sub & (deviation <= np.quantile(deviation[sub], 1.0 - trim))
            if keep.sum() > basis.shape[1]:
                coef, *_ = np.linalg.lstsq(basis[keep], values[keep], rcond=None)
        out[..., c] = (values - basis @ coef).reshape(h, w)
    return out


def _residual_magnitude(
    clean: Image.Image,
    edit: Image.Image,
    detrend_degree: int,
    detrend_trim: float,
    smooth_px: float,
    valid: np.ndarray | None,
) -> np.ndarray:
    """Smoothed magnitude of the detrended ``edit - clean`` residual (float, H x W).

    Detrending drops the editor's global photometric drift; the residual magnitude is then
    smoothed by ``smooth_px`` so per-pixel sensor noise does not fragment the mask.
    """
    c = np.asarray(clean.convert("RGB"), dtype=np.float64)
    e = np.asarray(edit.convert("RGB").resize(clean.size, _RESAMPLE), dtype=np.float64)
    diff = e - c
    if detrend_degree >= 0:
        diff = _detrend(diff, detrend_degree, detrend_trim, valid)
    return ndimage.gaussian_filter(np.sqrt((diff**2).sum(axis=2)), float(smooth_px))


def _lower_half_mad(ref: np.ndarray, centre: float) -> float:
    """MAD-equivalent sigma estimated from the pixels at or below ``centre`` only.

    Using one side makes the scale estimate independent of how much of the *upper* tail is
    defect, which is the whole point here. For an uncontaminated Gaussian background the
    lower-half MAD equals the full MAD, so this is unbiased in the normal case and merely
    robust in the contaminated one.
    """
    low = ref[ref <= centre]
    if low.size == 0:
        return 0.0
    return float(1.4826 * np.median(centre - low))


def _noise_level(
    mag: np.ndarray,
    valid: np.ndarray | None,
    *,
    bg_quantile: float = 0.20,
    clip_sigma: float = 3.0,
    clip_iters: int = 5,
    min_keep_frac: float = 0.15,
) -> tuple[float, float]:
    """Robust ``(background, sigma)`` of ``mag`` over ``valid``, defining the threshold.

    The obvious estimator — the median and MAD of everything in ``valid`` — tolerates the
    defect contaminating the sample only while the defect stays under half the region, and
    *silently inverts* past that point: the median moves inside the change, the threshold
    rises above the defect, and the result is an empty mask rather than an error. On the v3
    corpus that cost 812 of 5360 candidates their pixel ground truth, concentrated on exactly
    the severe texture cells the benchmark most needs (tile kept 6 of 100 ``severe``).

    This moves the breakdown point to a configurable fraction instead:

    1. Seed the background at ``bg_quantile`` instead of the median, which tolerates
       contamination up to roughly ``1 - bg_quantile``. Measured on a synthetic re-render
       covering a growing fraction of the crop, recall of the changed region is 1.00 up to
       ~65-70% at the 0.20 default and collapses to 0.00 beyond it, against ~50% for the plain
       median. Lowering it further extends that reach (0.10 holds to 75%) but reads the seed
       off a thinner slice of the distribution, which makes ``sigma`` noisier on the small
       ``valid`` regions the remask path produces. It has no effect at all on ordinary small
       defects — a 13 px pit gives a bit-identical mask at every value between 0.10 and 0.40 —
       so this parameter trades only large-defect reach.
    2. Take the scale from :func:`_lower_half_mad` about that seed, so however much of the
       upper tail is defect cannot inflate it.
    3. Refine: repeatedly drop everything above ``background + clip_sigma * sigma`` and
       re-estimate, which recovers the efficiency the quantile seed gives up when the sample
       is barely contaminated after all. Stops on convergence, at ``clip_iters``, or before
       the surviving sample falls below ``min_keep_frac``.

    Plain sigma clipping without the quantile seed does not work here: past 50% contamination
    the first median already sits inside the change and the inflated MAD puts the cut above
    every pixel, so the first iteration removes nothing and it converges straight back to the
    contaminated answer. The seed is what moves the breakdown point; the iteration only
    sharpens it.

    A spatially local (windowed) noise floor does not work either, and fails in the opposite
    direction: a window placed inside a large defect is *entirely* defect, so local calibration
    blinds the interior of exactly the defects this is meant to recover.
    """
    ref = mag if valid is None else mag[valid]
    if ref.size == 0:
        return 0.0, 0.0

    med = float(np.quantile(ref, float(bg_quantile)))
    sigma = _lower_half_mad(ref, med)
    floor = max(1, int(float(min_keep_frac) * ref.size))
    for _ in range(int(clip_iters)):
        if sigma <= 0.0:
            break  # a degenerate sample (synthetic image, all-zero residual): nothing to refine
        keep = ref[ref <= med + float(clip_sigma) * sigma]
        if keep.size < floor:
            break
        new_med = float(np.median(keep))
        new_sigma = _lower_half_mad(keep, new_med)
        if new_sigma <= 0.0:
            break
        converged = abs(new_med - med) <= 1e-9 and abs(new_sigma - sigma) <= 1e-9
        med, sigma = new_med, new_sigma
        if converged:
            break
    return med, sigma


def _keep_large_components(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, int]:
    """Drop connected components smaller than ``min_area_px``; return ``(mask, n_kept)``."""
    if not mask.any():
        return mask, 0
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    keep = np.flatnonzero(sizes >= max(1, min_area_px))
    return np.isin(labels, keep), int(keep.size)


def _mask_geometry(binary: np.ndarray) -> tuple[float, int]:
    """``(coverage, n_components)`` of a bool mask, JSON-safe Python scalars."""
    binary = np.asarray(binary, dtype=bool)
    n = int(ndimage.label(binary)[1]) if binary.any() else 0
    return round(float(binary.mean()), 6), n


def _json_scalar(value: object) -> object:
    """Coerce numpy scalars so the trace round-trips through JSONL."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def _mask_step(step: str, effect: str, binary: np.ndarray, **extra: object) -> dict:
    """One pipeline-stage record: name, what it did, geometry after, optional extras.

    ``effect`` is a short verb (``seeded``, ``pruned``, ``grew``, ``fallback``,
    ``skipped``, …). Keys with a ``None`` extra are omitted so a no-op does not
    pretend it had a count.
    """
    coverage, components = _mask_geometry(binary)
    rec: dict = {"step": step, "effect": effect, "coverage": coverage, "components": components}
    for key, value in extra.items():
        if value is None:
            continue
        rec[key] = _json_scalar(value)
    return rec


@dataclass(frozen=True)
class SweepStability:
    """Residual-threshold inclusion frequency and the derived QC scalar.

    ``freq`` is per-pixel inclusion over the k-grid in ``[0, 1]``. Core pixels have
    frequency 1 (survive every k); flicker pixels have frequency in ``(0, 1)``.
    ``uncertainty`` is ``area(flicker) / area(core)``, or ``None`` when the core is empty
    (JSON-safe; never ``inf``).
    """

    freq: np.ndarray
    uncertainty: float | None
    core_coverage: float
    flicker_coverage: float

    @property
    def core(self) -> np.ndarray:
        """Pixels included at every k."""
        return np.asarray(self.freq) >= _FREQ_ONE

    @property
    def flicker(self) -> np.ndarray:
        """Pixels included at some but not all k."""
        f = np.asarray(self.freq)
        return (f > 0.0) & (f < _FREQ_ONE)


def resolve_k_sweep(k_sweep: Sequence[float] | None, k_sigma: float) -> tuple[float, ...]:
    """Sorted unique k-grid, always including the operating ``k_sigma``.

    ``None`` or an empty sequence falls back to :data:`DEFAULT_K_SWEEP` (empty → operating k
    alone, so a caller can disable the sweep).
    """
    if k_sweep is None:
        raw: tuple[float, ...] = DEFAULT_K_SWEEP
    else:
        raw = tuple(float(k) for k in k_sweep)
        if not raw:
            raw = (float(k_sigma),)
    return tuple(sorted({float(k_sigma), *raw}))


def threshold_sweep_stability(
    mag: np.ndarray,
    med: float,
    sigma: float,
    *,
    ks: Sequence[float],
    min_change: float,
    min_area_px: int,
    valid: np.ndarray | None,
    operating: np.ndarray,
) -> SweepStability:
    """Inclusion frequency of seed masks over ``ks``, restricted to the defect band.

    Each k is thresholded the same way as the operating seed (``background + k·sigma``,
    floored at ``min_change``, ``valid``-masked, then ``min_area_px`` CC filter). After
    voting, connected components of ``{freq > 0}`` that do not intersect ``operating`` are
    dropped so distant low-k speckle does not inflate the scalar. Seed-threshold only: the
    sweep does not re-run refinement.
    """
    ks = tuple(float(k) for k in ks)
    n = len(ks)
    votes = np.zeros(mag.shape, dtype=np.float64)
    if n:
        for k in ks:
            threshold = max(med + k * sigma, float(min_change))
            hit = mag > threshold
            if valid is not None:
                hit &= valid
            hit, _ = _keep_large_components(hit, min_area_px)
            votes += hit.astype(np.float64)
        freq = votes / float(n)
    else:
        freq = np.zeros(mag.shape, dtype=np.float64)

    operating_b = np.asarray(operating, dtype=bool)
    support = freq > 0.0
    if support.any() and operating_b.any():
        labels, _ = ndimage.label(support)
        overlap = np.unique(labels[operating_b])
        overlap = overlap[overlap != 0]
        keep = np.isin(labels, overlap) if overlap.size else np.zeros_like(support)
        freq = np.where(keep, freq, 0.0)
    else:
        freq = np.zeros(mag.shape, dtype=np.float64)

    core = freq >= _FREQ_ONE
    flicker = (freq > 0.0) & ~core
    core_n = int(core.sum())
    flicker_n = int(flicker.sum())
    n_pix = float(mag.size)
    return SweepStability(
        freq=freq,
        uncertainty=(flicker_n / core_n) if core_n else None,
        core_coverage=core_n / n_pix,
        flicker_coverage=flicker_n / n_pix,
    )


def _dilate_feather(binary: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    """Compositing weight in [0,1]: grow ``binary`` by ``dilate_px + feather_px``, then feather.

    The margin keeps the editor's own soft defect boundary (shadow, debris halo) from being
    clipped at the mask edge, and the feather hides the composite seam.
    """
    f = binary.astype(np.float64)
    radius = int(dilate_px) + int(feather_px)
    if radius > 0:
        f = ndimage.maximum_filter(f, size=2 * radius + 1, mode="constant", cval=0.0)
    if feather_px > 0:
        f = np.clip(ndimage.gaussian_filter(f, float(feather_px)), 0.0, 1.0)
    return f


def _apply_guided_refine(
    seed: np.ndarray,
    raw: np.ndarray,
    *,
    dilate_r: int,
    core: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, str, dict]:
    """Band-constrain ``raw`` to ``seed`` and accept it, or fall back to the seed.

    Always intersect-constrained: snapping the coarse feature seed onto residual edges is
    meant to *drop* the halo, so the refined mask may shrink but can never leave
    ``dilate(seed, dilate_r)``. Acceptance is the extent range alone — the k-sweep core is
    reported for the audit but not gated on, because on a coarse seed that core *is* the
    blocky plateau this step exists to trim.
    """
    constrained, _ = intersect_constrain(seed, raw, dilate_r)
    cov = float(constrained.mean())
    core_b = np.asarray(core, dtype=bool)
    core_n = int(core_b.sum())
    core_recall = (
        1.0 if core_n == 0 else float(np.logical_and(constrained, core_b).sum()) / core_n
    )
    extra = dict(kind="guided", iou=binary_iou(constrained, seed), core_recall=core_recall)
    if bool(constrained.any()) and lo <= cov <= hi:
        new_n, seed_n = int(constrained.sum()), int(seed.sum())
        effect = "shrunk" if new_n < seed_n else "unchanged"
        return constrained, "guided", _mask_step("refine", effect, constrained, **extra)
    extra["proposed_coverage"] = cov
    return seed, "guided_fallback", _mask_step("refine", "fallback", seed, **extra)


def estimate_mask(
    clean: Image.Image,
    edit: Image.Image,
    *,
    semantic_fn: SemanticFn,
    stage: str | None = None,
    extent_ranges: dict[str, tuple[float, float]] | None = None,
    detrend_degree: int = 2,
    detrend_trim: float = 0.10,
    smooth_px: float = 3.0,
    k_sigma: float = 4.0,
    k_sweep: Sequence[float] | None = None,
    min_area_px: int = 64,
    dilate_px: int = 12,
    feather_px: int = 12,
    bg_quantile: float = 0.20,
    clip_sigma: float = 3.0,
    clip_iters: int = 5,
    valid: np.ndarray | None = None,
    dilate_r: int = 24,
    resnet_min_change: float = 0.02,
    guide_refine: bool = False,
    guide_radius: int = 24,
    guide_eps: float = 1e-4,
    seed_stage: str = SEED_STAGE,
) -> MaskResult:
    """Estimate the defect mask for one ``(clean, edit)`` crop pair.

    ``edit`` should already carry whatever photometric correction the composite applies, so
    that the mask describes the change actually present in the stored image;
    :func:`synevad.synthesis.blend.blend_candidate` arranges this by colour-matching before estimating.

    ``semantic_fn`` supplies the dense change map the mask is seeded from — DINOv3
    patch-token or ImageNet ResNet-18 feature-map cosine distance
    (:func:`synevad.synthesis.semantic.load_change_fns`). It is injected rather than constructed
    here so tests never load weights. ``seed_stage`` is recorded on the result
    (``dinov3`` or ``resnet18``).

    ``guide_refine`` snaps that seed onto the photometric residual via
    :func:`synevad.synthesis.refine.guided_refine`. The feature map still decides support; the residual
    places the edge. ``guide_radius`` defaults to 24 (about three layer2 cells) so the
    window reaches from the stair outer edge to the residual edge. This is what the
    post-blend pass runs; the paste seed does not.

    ``k_sigma`` sets the detection threshold in units of the change map's own noise floor,
    floored at ``resnet_min_change``. A crop the editor left unchanged yields nothing above
    it, so the result is :data:`STATUS_NO_DEFECT`. ``k_sweep`` (default
    :data:`DEFAULT_K_SWEEP`) is the threshold grid for inclusion-frequency uncertainty; it
    is always unioned with ``k_sigma``. The sweep does not re-run refinement — ``soft`` is
    seed-threshold stability, not stability of the final geometry.

    ``bg_quantile`` / ``clip_sigma`` / ``clip_iters`` tune how that noise floor is estimated;
    see :func:`_noise_level`. It tolerates a defect covering most of the crop, which a plain
    median+MAD does not — that inverts silently instead, returning an empty mask for the
    largest defects. The ``object_scale`` extent bucket is only reachable because of it.

    There is a limit no calibration passes. If the editor re-rendered the *entire* crop there
    is no unchanged population left in it to measure a noise floor against, and no estimator
    can separate defect from surface; such candidates stay rejected, and correctly so.

    ``valid`` (bool, crop-shaped) restricts the noise estimate and the mask to the region
    where the change is meaningful; pass it when re-masking an already-composited pair.

    A non-``ok`` status is a statement about *this candidate*, not an error: the caller is
    expected to persist it with the status recorded.
    """
    shape = (clean.size[1], clean.size[0])
    if valid is not None and valid.shape != shape:
        raise ValueError(f"valid mask {valid.shape} does not match crop {shape}")

    mag = np.asarray(semantic_fn(clean, edit), dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError(f"semantic_fn must return an H x W map, got shape {mag.shape}")
    if mag.shape != shape:
        mag = np.asarray(
            Image.fromarray(mag.astype(np.float32), mode="F").resize(
                clean.size, Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
    if smooth_px > 0:
        mag = ndimage.gaussian_filter(mag, float(smooth_px))
    mag_floor = float(resnet_min_change)
    med, sigma = _noise_level(
        mag,
        valid,
        bg_quantile=bg_quantile,
        clip_sigma=clip_sigma,
        clip_iters=clip_iters,
    )
    # Absolute threshold in noise units, floored at ``resnet_min_change`` in cosine/L2 units.
    # The floor matters whenever the scale collapses — a synthetic image, or a change map that
    # is identically zero over more than half the region (which is exactly what re-masking a
    # composited crop looks like outside the old alpha). Without it a vanishing sigma makes
    # every rounding wobble "significant".
    threshold = max(med + k_sigma * sigma, mag_floor)
    hit = mag > threshold
    if valid is not None:
        hit &= valid
    raw_n = int(ndimage.label(hit)[1]) if np.asarray(hit).any() else 0
    hit, components = _keep_large_components(hit, min_area_px)
    dropped = raw_n - components
    stability = threshold_sweep_stability(
        mag,
        med,
        sigma,
        ks=resolve_k_sweep(k_sweep, k_sigma),
        min_change=mag_floor,
        min_area_px=min_area_px,
        valid=valid,
        operating=hit,
    )

    extent_name = extent_for_stage(stage)
    ranges = extent_ranges or DEFAULT_EXTENT_RANGES
    lo, hi = ranges.get(extent_name, DEFAULT_EXTENT_RANGES["localized"])
    trace: list[dict] = [
        _mask_step(
            "seed",
            "empty" if components == 0 else "seeded",
            hit,
            source=seed_stage,
            dropped=dropped or None,
        )
    ]

    def pack(binary: np.ndarray, status: str, mask_stage: str) -> MaskResult:
        binary = np.asarray(binary, dtype=bool)
        n_comp = int(ndimage.label(binary)[1]) if binary.any() else 0
        coverage = float(binary.mean())
        alpha = _dilate_feather(binary, dilate_px, feather_px)
        to_img = lambda a: Image.fromarray(  # noqa: E731
            np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"
        )
        return MaskResult(
            soft=to_img(stability.freq),
            binary=to_img(binary.astype(np.float64)),
            alpha=to_img(alpha),
            coverage=coverage,
            components=n_comp,
            status=status,
            mask_stage=mask_stage,
            expected_extent=extent_name,
            extent_lo=float(lo),
            extent_hi=float(hi),
            mask_uncertainty=stability.uncertainty,
            mask_core_coverage=stability.core_coverage,
            mask_flicker_coverage=stability.flicker_coverage,
            trace=tuple([*trace, _mask_step("accept", status, binary, status=status)]),
        )

    if components == 0:
        # Nothing localised in the feature-map change: the editor left the crop alone, or
        # changed it so uniformly that no component stands out. Unusable as pixel GT either
        # way. (A photometric estimator could tell a wholesale re-render apart by the plain
        # residual median; a feature map never sees it, so the diagnosis is the same one.)
        return pack(hit, STATUS_NO_DEFECT, seed_stage)

    mask_stage = seed_stage
    seed = np.asarray(hit, dtype=bool)
    if guide_refine:
        residual_guide = _residual_magnitude(
            clean, edit, detrend_degree, detrend_trim, smooth_px, valid
        )
        raw = (
            guided_refine(
                seed.astype(np.float64),
                residual_guide,
                radius=int(guide_radius),
                eps=float(guide_eps),
            )
            >= 0.5
        )
        if valid is not None:
            raw = np.logical_and(raw, valid)
        hit, mask_stage, step = _apply_guided_refine(
            seed,
            raw,
            dilate_r=dilate_r,
            core=stability.core,
            lo=lo,
            hi=hi,
        )
        trace.append(step)

    coverage = float(np.asarray(hit, dtype=bool).mean())
    if coverage > hi:
        status = STATUS_OVER_COVERAGE
    elif coverage < lo:
        # Localised, but smaller than the stage claims. Kept apart from `over_coverage`
        # because the two say opposite things about the generator: too large means the editor
        # re-rendered rather than drew, too small means it drew but under-delivered on the
        # severity the prompt asked for. Both are unusable as GT *for the stage they are filed
        # under*, which is why neither is `ok`.
        status = STATUS_UNDER_COVERAGE
    else:
        status = STATUS_OK
    return pack(hit, status, mask_stage)


def alpha_support(
    binary: Image.Image, dilate_px: int, feather_px: int, level: float = 0.5
) -> np.ndarray:
    """Region where a composite built from ``binary`` carried the edit at weight > ``level``.

    Inverse of the compositing step, for re-masking: ``blended = alpha*edit + (1-alpha)*clean``,
    so only where alpha is appreciable does the stored crop carry the edit's own pixels.
    Outside, the crop is bit-identical to the clean one and its residual is exactly zero,
    which would collapse the noise estimate.

    ``level`` deliberately defaults to the half-weight contour rather than something near 1:
    feathering with a sigma comparable to the dilation means the weight only reaches 1.0 deep
    inside a large region, so a stricter contour would shrink — or empty — the usable region
    for exactly the small defects worth keeping.
    """
    f = _dilate_feather(np.asarray(binary.convert("L")) > 127, dilate_px, feather_px)
    return f > level


def mask_result_from_binary(
    binary: Image.Image,
    *,
    dilate_px: int = 12,
    feather_px: int = 12,
    mask_stage: str = "provided",
) -> MaskResult:
    """Wrap a caller-supplied binary mask as a :class:`MaskResult` for compositing.

    No extent gate: the mask is treated as author GT. Empty → ``no_defect``. ``alpha`` is
    the same dilated+feathered paste weight the estimator would have built from this
    binary. ``soft`` copies the binary (there is no k-sweep).
    """
    hit = np.asarray(binary.convert("L")) > 127
    coverage = float(hit.mean())
    n_comp = int(ndimage.label(hit)[1]) if hit.any() else 0
    alpha = _dilate_feather(hit, dilate_px, feather_px)

    def to_img(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"
        )

    status = STATUS_OK if hit.any() else STATUS_NO_DEFECT
    return MaskResult(
        soft=to_img(hit.astype(np.float64)),
        binary=to_img(hit.astype(np.float64)),
        alpha=to_img(alpha),
        coverage=coverage,
        components=n_comp,
        status=status,
        mask_stage=mask_stage,
        expected_extent=None,
        extent_lo=None,
        extent_hi=None,
        mask_uncertainty=None,
        mask_core_coverage=coverage,
        mask_flicker_coverage=0.0,
        trace=(),
    )


# --- placement + persistence ----------------------------------------------


OVERLAY_COLOR = (214, 69, 47)
OVERLAY_OPACITY = 0.5


def place_mask(
    mask: Image.Image, box: Box, full_size: tuple[int, int], *, nearest: bool = False
) -> Image.Image:
    """Place a crop ``mask`` into a black full-frame canvas (L) at ``box`` (mirrors the composite)."""
    w, h = full_size
    x0, y0, x1, y1 = box
    canvas = Image.new("L", (w, h), 0)
    resample = Image.Resampling.NEAREST if nearest else _RESAMPLE
    canvas.paste(mask.convert("L").resize((x1 - x0, y1 - y0), resample), (x0, y0))
    return canvas


def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    *,
    color: tuple[int, int, int] = OVERLAY_COLOR,
    opacity: float = OVERLAY_OPACITY,
) -> Image.Image:
    """Tint ``image`` where the binary ``mask`` is on. Off-mask pixels are unchanged.

    ``mask`` is treated as L (threshold 127). If its size differs from ``image``, it is
    nearest-neighbour resized. An empty mask returns an RGB copy of ``image``.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    m = np.asarray(mask.convert("L"))
    if m.shape != rgb.shape[:2]:
        m = np.asarray(
            mask.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        )
    on = m > 127
    if opacity <= 0.0 or not on.any():
        return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    tint = np.array(color, dtype=np.float64)
    rgb[on] = (1.0 - opacity) * rgb[on] + opacity * tint
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def write_masks(
    result: MaskResult,
    box: Box,
    full_size: tuple[int, int],
    *,
    mask_crop_dir: Path,
    mask_soft_crop_dir: Path,
    mask_full_dir: Path,
    mask_soft_full_dir: Path,
    name: str,
    overlay_crop_dir: Path | None = None,
    overlay_full_dir: Path | None = None,
    patch: Image.Image | None = None,
    full: Image.Image | None = None,
) -> dict[str, str]:
    """Write all four mask variants for one candidate and return their resolved paths.

    The crop-level binary and soft masks, plus both placed into a full-frame canvas aligned
    with the composite (binary crisp via nearest, soft interpolated). A non-``ok`` result is
    still written — as the empty or over-covered mask it actually is — so the candidate keeps
    a complete record and is excluded downstream by its ``mask_status``, not by a missing file.

    When ``overlay_crop_dir`` / ``overlay_full_dir`` are set, also write RGB overlays of
    the binary GT on ``patch`` / ``full`` (the stored composites). Those images are
    required in that case.
    """
    paths = {
        "mask_path": mask_crop_dir / f"{name}.png",
        "mask_soft_path": mask_soft_crop_dir / f"{name}.png",
        "mask_full_path": mask_full_dir / f"{name}.png",
        "mask_soft_full_path": mask_soft_full_dir / f"{name}.png",
    }
    result.binary.save(paths["mask_path"])
    result.soft.save(paths["mask_soft_path"])
    placed_binary = place_mask(result.binary, box, full_size, nearest=True)
    placed_binary.save(paths["mask_full_path"])
    place_mask(result.soft, box, full_size, nearest=False).save(paths["mask_soft_full_path"])
    if overlay_crop_dir is not None:
        if patch is None:
            raise ValueError("overlay_crop_dir requires patch")
        paths["mask_overlay_path"] = overlay_crop_dir / f"{name}.png"
        overlay_mask(patch, result.binary).save(paths["mask_overlay_path"])
    if overlay_full_dir is not None:
        if full is None:
            raise ValueError("overlay_full_dir requires full")
        paths["mask_overlay_full_path"] = overlay_full_dir / f"{name}.png"
        overlay_mask(full, placed_binary).save(paths["mask_overlay_full_path"])
    return {k: str(v.resolve()) for k, v in paths.items()}
