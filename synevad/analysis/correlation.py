"""How well the synthetic benchmark stands in for the real one.

`proxy_quality` is the single-population statistic: given the real and synthetic scores
of the same set of models, how strongly does one predict the other. `correlate` applies
it per group of an already-paired frame.

Neither function knows where the numbers came from, and neither ever emits a warning —
degenerate inputs (too few points, all-NaN, zero variance) return NaN fields instead, so
a scan over a few hundred groups stays readable.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

# Fixed key set of `proxy_quality`, so `pd.DataFrame.from_records` gets a stable schema
# no matter which branch each group took.
PROXY_QUALITY_FIELDS: tuple[str, ...] = (
    "n",
    "n_dropped",
    "alpha",
    "pearson_r",
    "pearson_p",
    "pearson_r2",
    "pearson_ci_lo",
    "pearson_ci_hi",
    "spearman_rho",
    "spearman_p",
    "kendall_tau",
    "kendall_p",
    "real_min",
    "real_max",
    "real_range",
    "real_std",
    "synth_min",
    "synth_max",
    "synth_range",
    "synth_std",
)

_NAN = float("nan")


def _spread(values: np.ndarray) -> dict[str, float]:
    """Min/max/range/std of one arm. `range` is the range-restriction alarm: a real
    metric at ceiling has nothing left for the synthetic one to track."""
    if len(values) == 0:
        return {"min": _NAN, "max": _NAN, "range": _NAN, "std": _NAN}
    lo, hi = float(values.min()), float(values.max())
    std = float(values.std(ddof=1)) if len(values) > 1 else _NAN
    return {"min": lo, "max": hi, "range": hi - lo, "std": std}


def proxy_quality(
    real: ArrayLike, synth: ArrayLike, *, alpha: float = 0.05, min_n: int = 4
) -> dict[str, float]:
    """Correlation between a real metric and its synthetic counterpart across models.

    Pearson measures linearity/calibration, Spearman rank agreement, Kendall (tau-b, so
    tie-corrected — the real metric is often near-constant) a conservative rank estimate.
    `pearson_ci_*` is a Fisher-z interval at `alpha`.

    Non-finite pairs are dropped pairwise and counted in `n_dropped`; `aupro` in
    particular is missing for whole backbones, and an all-NaN result would otherwise read
    as "no correlation" rather than "no data". Below `min_n` points, or when either arm
    is constant, the correlation fields are NaN: scipy raises at n=1, returns a
    meaningless r=1 at n=2, and the Fisher-z standard error divides by zero at n=3.
    """
    real = np.asarray(real, dtype=float).ravel()
    synth = np.asarray(synth, dtype=float).ravel()
    if real.shape != synth.shape:
        raise ValueError(
            f"real and synth differ in length: {real.size} vs {synth.size}"
        )

    keep = np.isfinite(real) & np.isfinite(synth)
    n_dropped = int(keep.size - keep.sum())
    real, synth = real[keep], synth[keep]
    n = int(real.size)

    real_spread, synth_spread = _spread(real), _spread(synth)
    result: dict[str, float] = {
        "n": n,
        "n_dropped": n_dropped,
        "alpha": alpha,
        "pearson_r": _NAN,
        "pearson_p": _NAN,
        "pearson_r2": _NAN,
        "pearson_ci_lo": _NAN,
        "pearson_ci_hi": _NAN,
        "spearman_rho": _NAN,
        "spearman_p": _NAN,
        "kendall_tau": _NAN,
        "kendall_p": _NAN,
        **{f"real_{key}": val for key, val in real_spread.items()},
        **{f"synth_{key}": val for key, val in synth_spread.items()},
    }

    # Pre-check instead of letting scipy warn: a constant arm is routine here, since a
    # real metric can sit at 1.0 for every model in the sweep.
    degenerate = real_spread["std"] == 0.0 or synth_spread["std"] == 0.0
    if n < min_n or degenerate:
        return result

    pearson = stats.pearsonr(synth, real)
    spearman = stats.spearmanr(synth, real)
    kendall = stats.kendalltau(synth, real)

    r = float(pearson.statistic)  # type: ignore (these attributes are available)
    result["pearson_r"] = r
    result["pearson_p"] = float(pearson.pvalue)  # type: ignore
    result["pearson_r2"] = r**2
    result["spearman_rho"] = float(spearman.statistic)  # type: ignore
    result["spearman_p"] = float(spearman.pvalue)  # type: ignore
    result["kendall_tau"] = float(kendall.statistic)  # type: ignore
    result["kendall_p"] = float(kendall.pvalue)  # type: ignore

    # arctanh(±1) is infinite, which would collapse the interval onto the estimate and
    # report perfect certainty; leave it undefined instead.
    if n > 3 and np.isfinite(r) and abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        zc = stats.norm.ppf(1 - alpha / 2)
        result["pearson_ci_lo"] = float(np.tanh(z - zc * se))
        result["pearson_ci_hi"] = float(np.tanh(z + zc * se))

    return result


def correlate(
    paired: pd.DataFrame,
    by: Sequence[str] = ("metric", "query", "category"),
    *,
    alpha: float = 0.05,
    min_n: int = 4,
    min_defect_samples: int | None = None,
) -> pd.DataFrame:
    """`proxy_quality` per group of `by`, one row per group.

    `paired` is a `build_paired_metrics` frame; each group is a population of models
    whose `real` and `synth` columns get correlated. Widening `by` splits populations,
    dropping a key pools them — `by=("metric", "query")` pools categories, for instance.

    `min_defect_samples` screens out queries that selected too few defects to score,
    mirroring `config.min_num_defects`; off by default so the screening stays visible in
    the caller.
    """
    by = list(by)
    missing = [key for key in by if key not in paired.columns]
    if missing:
        raise KeyError(
            f"group keys {missing} not in paired frame; available: {list(paired.columns)}"
        )

    if min_defect_samples is not None:
        paired = paired[paired["n_defect_samples"] >= min_defect_samples]

    if paired.empty:
        return pd.DataFrame(columns=[*by, *PROXY_QUALITY_FIELDS])

    rows = []
    # dropna=False: a group key may legitimately be null (older runs are missing some
    # params), and the pandas default would silently drop those populations.
    for keys, group in paired.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            dict(
                zip(by, keys),
                **proxy_quality(
                    group["real"], group["synth"], alpha=alpha, min_n=min_n
                ),
            )
        )

    return pd.DataFrame.from_records(rows).astype({"n": "int64", "n_dropped": "int64"})
