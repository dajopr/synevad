"""How much of a frame a defect covers — the one mask statistic the eval gate needs.

Every synthetic arm gates on ``defect_area_frac != 0``: a composite whose mask is empty is
label noise, because the image it produced is bit-identical to the clean frame it was
edited from, and scoring it as an anomaly asks the detector to find something that is not
there. Measuring it needs nothing but the mask.

``!= 0`` rather than ``> 0`` on purpose. A row that was never scored carries NaN, and in
the pandas query the gate is written as, ``NaN != 0`` is True while ``NaN > 0`` is False —
so ``!=`` keeps unscored rows and drops only *measured*-empty ones, which is the intent.
:func:`passes_eval_gate` in ``synevad.data.synevad`` is the same rule in Python.
"""

from __future__ import annotations

import numpy as np


def _as_bool_mask(mask: np.ndarray | object) -> np.ndarray:
    """Accept 0/1, 0/255, bool or float masks; anything above half-scale is defect."""
    if not isinstance(mask, np.ndarray) and hasattr(mask, "convert"):
        mask = mask.convert("L")
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == bool:
        return array
    threshold = 0.5 if np.issubdtype(array.dtype, np.floating) else (array.max() or 1) / 2
    return array > threshold


def defect_area_fraction(mask: np.ndarray | object) -> float:
    """``|defect| / |frame|`` — the share of the frame the mask covers, in ``[0, 1]``.

    Exactly 0 means a measured-empty mask, which is the sentinel the eval gate drops; it
    is a real measurement, not a missing one, so it must not be confused with NaN.
    """
    binary = _as_bool_mask(mask)
    return float(binary.sum()) / float(binary.size) if binary.size else float("nan")


def area_fields(mask: np.ndarray | object) -> dict[str, float]:
    """The manifest columns a generator writes beside each composite.

    One helper so every generation path — FLUX editing, DRAEM overlays, any future arm —
    spells the column the same way and the eval gate can rely on it being there.
    """
    return {"defect_area_frac": round(defect_area_fraction(mask), 6)}
