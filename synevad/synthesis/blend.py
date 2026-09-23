"""Composite a FLUX edit back into the source frame (and crop), with seam-hiding.

Self-contained blend primitives (numpy/scipy/Pillow, no model).
:func:`blend_candidate` owns everything after the edit: it asks :mod:`synevad.synthesis.masks` for the
defect mask, composites only the masked defect over the *original* surface, then
colour-matches and grains the patch. An empty mask (``no_defect``) short-circuits so the
stored crop is bit-identical to the clean crop.

The mask estimator and the colour-match are deliberately coupled: the estimator runs on
the colour-matched edit, so the mask describes the change actually present in the
stored image. :func:`blend_candidate` wires that up by colour-matching before estimating.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from synevad.synthesis.crops import Box, crop_image
from synevad.synthesis.masks import (
    DEFAULT_EXTENT_RANGES,
    DEFAULT_K_SWEEP,
    MaskResult,
    _dilate_feather,
    estimate_mask,
    with_post_blend,
)
from synevad.synthesis.semantic import default_min_change, resolve_encoder_kind, seed_block

_RESAMPLE = Image.Resampling.LANCZOS


# --- low-frequency helpers -------------------------------------------------


def color_match(
    edit: Image.Image, reference: Image.Image, strength: float = 1.0, sigma_frac: float = 0.25
) -> Image.Image:
    """Conform ``edit``'s low-frequency colour/lighting to ``reference`` while keeping detail.

    Subtracts FLUX's global exposure/white-balance shift via a blurred difference against the
    clean crop, so the slowly-varying tint is matched but the defect's high-frequency detail
    survives. ``edit`` and ``reference`` must be the same square size.
    """
    if strength <= 0:
        return edit
    e = np.asarray(edit.convert("RGB"), dtype=np.float64)
    r = np.asarray(reference.convert("RGB").resize(edit.size, _RESAMPLE), dtype=np.float64)
    sigma = sigma_frac * min(e.shape[:2])
    correction = ndimage.gaussian_filter(r, (sigma, sigma, 0.0)) - ndimage.gaussian_filter(
        e, (sigma, sigma, 0.0)
    )
    out = np.clip(e + strength * correction, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# --- grain + feather + paste ----------------------------------------------


def _noise_sigma(arr: np.ndarray) -> np.ndarray:
    """Per-channel noise std of an (H, W, 3) float array (Immerkær 1996, no SciPy)."""
    a = arr.astype(np.float64)
    lap = (
        4.0 * a[1:-1, 1:-1]
        - 2.0 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
        + (a[:-2, :-2] + a[:-2, 2:] + a[2:, :-2] + a[2:, 2:])
    )
    return np.sqrt(np.pi / 2.0) * np.abs(lap).mean(axis=(0, 1)) / 6.0


def match_grain(
    patch: Image.Image, reference: Image.Image, strength: float = 1.0, seed: int = 0
) -> Image.Image:
    """Add fresh grain to ``patch`` so its noise level matches ``reference`` (FLUX denoises)."""
    if strength <= 0:
        return patch
    p = np.asarray(patch.convert("RGB"), dtype=np.float64)
    r = np.asarray(reference.convert("RGB").resize(patch.size, _RESAMPLE), dtype=np.float64)
    add_sigma = np.sqrt(np.clip(_noise_sigma(r) ** 2 - _noise_sigma(p) ** 2, 0.0, None)) * strength
    grain = np.random.default_rng(seed).standard_normal(p.shape) * add_sigma
    return Image.fromarray(np.clip(p + grain, 0.0, 255.0).astype(np.uint8), mode="RGB")


def feather_mask(size: tuple[int, int], feather_px: int) -> np.ndarray:
    """Soft alpha (H, W) in [0,1] ramping from 0 at the border to 1 ``feather_px`` inside."""
    w, h = size
    if feather_px <= 0:
        return np.ones((h, w), dtype=np.float64)
    ys = np.minimum(np.arange(h), np.arange(h)[::-1])
    xs = np.minimum(np.arange(w), np.arange(w)[::-1])
    dist = np.minimum(ys[:, None], xs[None, :]).astype(np.float64)
    return np.clip(dist / float(feather_px), 0.0, 1.0)


def paste_crop(
    img: Image.Image,
    crop: Image.Image,
    box: Box,
    *,
    reference: Image.Image | None = None,
    feather_px: int = 0,
    color_strength: float = 1.0,
    color_sigma_frac: float = 0.25,
    grain_strength: float = 0.0,
    grain_seed: int = 0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Paste ``crop`` back into a copy of ``img`` at ``box`` with optional colour/grain/mask blend.

    ``reference`` enables colour-match; ``grain_strength`` adds sensor grain; ``mask`` (a defect
    alpha) composites only the defect over the true original pixels (replaces ``feather_px``);
    else ``feather_px`` alpha-composites the whole crop with a rectangular feather.
    """
    x0, y0, x1, y1 = box
    if reference is not None:
        crop = color_match(crop, reference, color_strength, color_sigma_frac)
    resized = crop.resize((x1 - x0, y1 - y0), _RESAMPLE)
    if grain_strength > 0:
        resized = match_grain(resized, img.crop(box), grain_strength, grain_seed)
    out = img.copy()
    if mask is None and feather_px <= 0:
        out.paste(resized, (x0, y0))
        return out
    base = np.asarray(out.convert("RGB"), dtype=np.float64)
    patch = np.asarray(resized.convert("RGB"), dtype=np.float64)
    if mask is not None:
        alpha = np.asarray(mask.convert("L").resize((x1 - x0, y1 - y0), _RESAMPLE), dtype=np.float64) / 255.0
        alpha = alpha[..., None]
    else:
        alpha = feather_mask((x1 - x0, y1 - y0), feather_px)[..., None]
    region = base[y0:y1, x0:x1]
    base[y0:y1, x0:x1] = alpha * patch + (1.0 - alpha) * region
    return Image.fromarray(base.astype(np.uint8), mode="RGB")


# --- orchestrator ----------------------------------------------------------


def resolve_extent_ranges(
    masks: dict, category: str | None = None
) -> dict[str, tuple[float, float]]:
    """Extent buckets for ``category``: ``masks.extent`` with ``masks.extent_overrides`` on top.

    The estimator gates a mask's area fraction against the range its *severity stage* maps to
    (:data:`synevad.synthesis.masks.STAGE_TO_EXTENT`). That is one of the two dimensions the ceiling varies
    along; the other is the category. A severe defect on some categories genuinely covers more
    of its crop than a chip on a hazelnut does, so a single set of buckets either rejects real
    defects or waves through wholesale re-renders.

    Overrides are per-bucket and *partial* — naming only ``object_scale`` for a category leaves
    its other two buckets at the defaults. The map is keyed by category rather than split
    texture/object because the categories that lose masks do not respect that split:
    metal_nut, pill and toothbrush are objects and are among the worst hit.
    """
    block = masks.get("extent") or {}
    per_cat = (masks.get("extent_overrides") or {}).get(category) or {} if category else {}
    out: dict[str, tuple[float, float]] = {}
    for name, default in DEFAULT_EXTENT_RANGES.items():
        raw = per_cat.get(name) or block.get(name, default)
        out[name] = (float(raw[0]), float(raw[1]))
    return out


def resolve_min_change(block: dict | None, default: float) -> float:
    """``block.min_change``, or ``default`` when the block omits it.

    Per-category values belong in ``synevad/synthesis/configs/category_overrides.yaml``, merged
    onto the config before this runs — not in a side map on the block.
    """
    block = block or {}
    if block.get("min_change") is None:
        return float(default)
    return float(block["min_change"])


def mask_params(masks: dict, category: str | None = None) -> dict:
    """Resolve :func:`synevad.synthesis.masks.estimate_mask` keyword arguments from the ``masks`` config.

    The ``extent`` / encoder sub-blocks, the per-category extent overrides, and the
    noise-floor keys are optional and read with defaults. The keys indexed directly stay
    required — a typo in one of those should fail loudly rather than silently fall back
    to a default that quietly changes the ground truth.
    """
    kind = resolve_encoder_kind(masks)
    return {
        "extent_ranges": resolve_extent_ranges(masks, category),
        "detrend_degree": int(masks["detrend_degree"]),
        "detrend_trim": float(masks["detrend_trim"]),
        "smooth_px": float(masks["smooth_px"]),
        "k_sigma": float(masks["k_sigma"]),
        "k_sweep": (
            tuple(float(k) for k in masks["k_sweep"])
            if masks.get("k_sweep") is not None
            else DEFAULT_K_SWEEP
        ),
        "min_area_px": int(masks["min_area_px"]),
        "dilate_px": int(masks["dilate_px"]),
        "feather_px": int(masks["alpha_feather_px"]),
        "bg_quantile": float(masks.get("bg_quantile", 0.20)),
        "clip_sigma": float(masks.get("clip_sigma", 3.0)),
        "clip_iters": int(masks.get("clip_iters", 5)),
        "dilate_r": int((masks.get("post_blend") or {}).get("dilate_r", 24)),
        "resnet_min_change": resolve_min_change(
            seed_block(masks), default_min_change(kind)
        ),
        "seed_stage": kind,
    }


def post_blend_enabled(masks: dict) -> bool:
    """True when a second encoder pass should measure the stored composite."""
    return bool((masks.get("post_blend") or {}).get("enabled", False))


def post_blend_mask_params(masks: dict, category: str | None = None) -> dict | None:
    """Estimator kwargs for the post-paste pass, or ``None`` when disabled.

    Inherits the first-pass knobs (k_sigma, extent, noise floor, …) then overlays any keys
    set in ``masks.post_blend``. ``guided`` snaps that seed onto the photometric
    residual so pixel GT is not stuck on the feature grid.
    """
    block = masks.get("post_blend") or {}
    if not bool(block.get("enabled", False)):
        return None
    kind = resolve_encoder_kind(masks)
    params = mask_params(masks, category)
    # Do not inherit the first-pass floor; post-blend has its own default.
    params["resnet_min_change"] = resolve_min_change(
        block, default_min_change(kind, post_blend=True)
    )
    params["seed_stage"] = kind
    params["guide_refine"] = bool(block.get("guided", False))
    params["guide_radius"] = (
        int(block["guide_radius"]) if block.get("guide_radius") is not None else 24
    )
    if block.get("guide_eps") is not None:
        params["guide_eps"] = float(block["guide_eps"])
    overrides = {
        "k_sigma": float,
        "smooth_px": float,
        "min_area_px": int,
        "bg_quantile": float,
        "clip_sigma": float,
        "clip_iters": int,
        "dilate_px": int,
        "dilate_r": int,
        "detrend_degree": int,
        "detrend_trim": float,
    }
    for key, cast in overrides.items():
        if key in block and block[key] is not None:
            params[key] = cast(block[key])
    if block.get("k_sweep") is not None:
        params["k_sweep"] = tuple(float(k) for k in block["k_sweep"])
    if "alpha_feather_px" in block and block["alpha_feather_px"] is not None:
        params["feather_px"] = int(block["alpha_feather_px"])
    return params


def blend_candidate(
    source_img: Image.Image,
    clean_crop: Image.Image,
    edit: Image.Image,
    box: Box,
    *,
    blending: dict,
    masks: dict,
    seed: int,
    resize_to: int | tuple[int, int],
    semantic_fn,
    post_semantic_fn=None,
    stage: str | None = None,
    category: str | None = None,
) -> tuple[Image.Image, Image.Image, MaskResult]:
    """Composite one edit into the frame and the crop, returning the mask result too.

    Estimates the defect mask on the colour-matched edit and composites through its alpha
    when the mask is non-empty, inserting only the defect over the original surface.
    When the mask is empty (``no_defect``) the source frame and clean crop are returned
    **unchanged** — bit-identical to ``clean_crop`` for the patch. The candidate is still
    persisted: the protocol needs every candidate on disk. Only its pixel ground truth is
    unusable, which is what the result's ``status`` records so downstream consumers can
    filter on it.

    ``semantic_fn`` is the dense change map that seeds the paste mask (DINOv3 or
    ResNet-18). ``post_semantic_fn`` is a second one run on ``(clean, blended)`` after
    the paste when ``masks.post_blend.enabled`` is set — that measurement becomes pixel
    GT; the first-pass alpha is still what was composited. ``masks.post_blend.guided``
    then snaps the seed onto the photometric residual. Both are injected so tests never
    load weights.

    ``stage`` picks the expected-extent bucket and ``category`` picks that bucket's per-category
    override when the config defines one; without either, every candidate is gated against the
    ``localized`` defaults.

    Returns ``(full_blended, blended_patch, mask_result)``.
    """
    matched = color_match(
        edit, clean_crop, blending["color_strength"], blending["color_sigma_frac"]
    )
    result = estimate_mask(
        clean_crop,
        matched,
        stage=stage,
        semantic_fn=semantic_fn,
        **mask_params(masks, category),
    )
    if result.coverage <= 0.0:
        return source_img.copy(), clean_crop.copy(), result

    full = paste_crop(
        source_img,
        matched,
        box,
        reference=None,
        feather_px=blending["feather_px"],
        grain_strength=blending["grain_strength"],
        grain_seed=blending["grain_seed"] + seed,
        mask=result.alpha,
    )
    blended_patch = crop_image(full, box, resize_to)
    post_params = post_blend_mask_params(masks, category)
    if post_params is not None:
        if post_semantic_fn is None:
            raise ValueError("post_blend.enabled requires post_semantic_fn")
        block = masks.get("post_blend") or {}
        valid = None
        if bool(block.get("constrain_to_alpha", True)):
            valid = np.asarray(result.alpha.convert("L")) > 127
        measured = estimate_mask(
            clean_crop,
            blended_patch,
            stage=stage,
            semantic_fn=post_semantic_fn,
            valid=valid,
            **post_params,
        )
        result = with_post_blend(
            result, measured, post_stage=f"{resolve_encoder_kind(masks)}_post"
        )
    return full, blended_patch, result


# Manifest `mask_status` of a negative control. Not `ok`: there is no defect, so the stored
# binary mask is empty by construction, and the eval labels these rows defect-free.
STATUS_NEGATIVE = "negative"


def blend_region(
    source_img: Image.Image,
    clean_crop: Image.Image,
    edit: Image.Image,
    box: Box,
    region: Image.Image,
    *,
    blending: dict,
    masks: dict,
    seed: int,
    resize_to: int | tuple[int, int],
) -> tuple[Image.Image, Image.Image, MaskResult, dict]:
    """Composite a negative-control inpaint through a *pre-chosen* region.

    The counterpart of :func:`blend_candidate` for negatives, and deliberately the same blend
    after the alpha is known: colour-match to the clean crop, paste through an alpha grown
    from the region by ``masks.dilate_px`` / ``masks.alpha_feather_px`` (exactly how a defect
    mask becomes its paste alpha), grain with the same seed offset. The only thing that differs
    from a defect is where the alpha came from and what the pixels under it show.

    The returned :class:`MaskResult` carries an **empty** binary — a negative has no defect
    pixels — with the region's alpha as the paste weight and ``status`` ``negative``. The stats
    dict records the region's coverage, the alpha's coverage, and ``region_change``: the mean
    per-pixel change the composite actually carries inside the alpha, in 8-bit levels, so a
    negative the model left untouched is visible instead of silently weakening the control.
    """
    binary = np.asarray(region.convert("L").resize(clean_crop.size, Image.Resampling.NEAREST)) > 127
    matched = color_match(
        edit, clean_crop, blending["color_strength"], blending["color_sigma_frac"]
    )
    alpha_arr = _dilate_feather(
        binary, int(masks["dilate_px"]), int(masks["alpha_feather_px"])
    )
    alpha = Image.fromarray(np.clip(alpha_arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    full = paste_crop(
        source_img,
        matched,
        box,
        reference=None,
        feather_px=blending["feather_px"],
        grain_strength=blending["grain_strength"],
        grain_seed=blending["grain_seed"] + seed,
        mask=alpha,
    )
    blended_patch = crop_image(full, box, resize_to)
    support = alpha_arr > 0.5
    diff = np.abs(
        np.asarray(blended_patch.convert("RGB"), dtype=np.float64)
        - np.asarray(clean_crop.convert("RGB").resize(blended_patch.size, _RESAMPLE), dtype=np.float64)
    ).mean(axis=2)
    support_patch = (
        np.asarray(alpha.resize(blended_patch.size, Image.Resampling.NEAREST)) > 127
    )
    stats = {
        "region_coverage": round(float(binary.mean()), 6),
        "region_alpha_coverage": round(float(support.mean()), 6),
        "region_change": round(float(diff[support_patch].mean()), 4) if support_patch.any() else 0.0,
    }
    empty = Image.new("L", clean_crop.size, 0)
    result = MaskResult(
        soft=empty,
        binary=empty,
        alpha=alpha,
        coverage=0.0,
        components=0,
        status=STATUS_NEGATIVE,
        mask_stage="region",
        paste_status=STATUS_NEGATIVE,
        paste_coverage=stats["region_alpha_coverage"],
        paste_stage="region",
        paste_components=int(ndimage.label(binary)[1]) if binary.any() else 0,
    )
    return full, blended_patch, result, stats
