"""Whether the synthetic arm reports a model as *better or worse* than it really is.

`correlation.py` asks whether the two arms co-vary and `selection.py` whether the
synthetic ranking picks a good model. Both are invariant to a shift: add 0.1 to every
synthetic score and the correlation, the ranking and the regret are all unchanged. So
neither can answer the question a reader asks first — *is the number the synthetic set
reports the model's real quality, or a flattering one?*

That question has a one-line answer, the signed gap

    bias = mean over models of (synth - real)

positive when the generated images make models look better than the real test set says
they are, negative when they make them look worse. `quality_bias` is that number for one
population, with the interval and the paired t-test that say whether its sign survives
the model-to-model spread; `evaluate_bias` applies it per group, mirroring `correlate`.

Read `bias` next to `bias_sd`, never alone
-----------------------------------------
A *constant* offset costs selection nothing — every model is flattered equally, the
ranking is untouched, and `skill` will happily be 1.0 while `bias` is +0.2. What breaks
selection is an offset that varies by model, and that is `bias_sd`. The pair reads:

    large bias, small bias_sd    the synthetic set is on a different scale, but usable
                                 for ranking; do not quote its AUROC as an estimate.
    small bias, large bias_sd    unbiased on average and wrong per model — the worst
                                 case, and the one a mean alone hides.

`overestimates` is the headline: True when the whole interval sits above zero, False when
it sits below, and null when it straddles — the honest third answer, which a bare sign
would hide. `over_share` is the same read without the normality assumption: the fraction
of models the synthetic arm flattered.

The axes: severity and scorer, not `query`
------------------------------------------
The sweep scores each synthetic set through the `queries` block of the config, whose
names decompose into two independent axes — a severity grade and, if present, a scorer
decision. `with_severity` splits them, so a hand-added `minimal_accept` is `minimal` x
`accept` rather than a third query that pools models already counted elsewhere. Queries
that do not decompose keep their own name as the severity, so an unrecognised one
becomes a visible extra row rather than silently joining the pooled cell. Default
configs only name the four grades and `all`.

What this must not be run on
----------------------------
The gap is only a calibration error when both arms are the *same* metric. On the
`PROXY_ONLY_METRICS` rows the real arm is a broadcast `image_auroc` standing in for a
metric real defects cannot have (see `synevad.analysis.data`), so `severity_cindex - real`
subtracts an AUROC from a concordance and means nothing. Those rows are dropped by
default rather than reported as a bias of 0.31.

Like the rest of the package, nothing here warns: degenerate populations return NaN
fields so a scan over a few hundred groups stays readable.

One caveat on the interval. The models in a population are a designed sweep, not a random
sample from anything, so `bias_ci_*` prices "does this offset hold across the configs that
were tried" and not a population inference. It is a spread, read as one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from synevad.metrics.ordinal import (
    SEVERITY_NAMES,
    STANDALONE_ORDER,
    severity_scheme,
)

from .data import PROXY_ONLY_METRICS

# The value both derived axes take when a query does not restrict them: `all` is the
# unfiltered query, and it is deliberately the same token the config already uses.
POOLED = "all"

# Scorer decisions a query name can carry (`config.queries`, `synevad.data.ScorerResult`).
# `none` is excluded on purpose — it is the grade a defect-free image carries, and it
# would collide with the severity axis.
SCORER_NAMES: frozenset[str] = frozenset({"accept", "reject"})

# Tokens that name no restriction at all, so `all` and `all_accept` land in the same cell.
_POOLED_TOKENS: frozenset[str] = frozenset({POOLED})

# The two axes `query` decomposes into. Appended to any `by` that the caller left the
# `query` key out of — without them the statistic is averaged over the very axes the
# question names.
BIAS_AXES: tuple[str, ...] = ("severity", "scorer")

DEFAULT_BIAS_BY: tuple[str, ...] = (
    "metric",
    "set_name",
    "scorer",
    "category",
    "severity",
)

# Fixed key set of `quality_bias`, so `pd.DataFrame.from_records` gets a stable schema no
# matter which branch each group took.
BIAS_FIELDS: tuple[str, ...] = (
    "n",
    "n_dropped",
    "alpha",
    "bias",
    "bias_median",
    "bias_sd",
    "bias_se",
    "bias_ci_lo",
    "bias_ci_hi",
    "bias_t",
    "bias_p",
    "abs_bias",
    "max_abs_bias",
    "over_share",
    "real_mean",
    "synth_mean",
    "overestimates",
)

_NAN = float("nan")

# A gap that is the same for every model up to floating point has no spread to test, but
# `std(ddof=1)` still returns ~1e-17 for it. Handing that to the t-test divides by it,
# warns about catastrophic cancellation and reports a t of 1e15. Anything below this
# fraction of the gap's own size is read as no spread at all — which is the interesting
# case, not a failure: a constant offset is exactly what leaves the ranking intact.
_SPREAD_EPS = 1e-9


def split_query(name) -> tuple[str, str]:
    """`(severity, scorer)` for one query name, e.g. `minimal_accept -> ('minimal', 'accept')`.

    Underscore-separated tokens are matched against the two vocabularies. A name whose
    tokens are not all recognised — a hand-added query with its own filter — is *not*
    decomposed: it keeps its whole name as the severity, so it shows up as its own row
    instead of quietly pooling with `all` and double-counting those models.
    """
    text = str(name).strip().lower()
    tokens = [token for token in text.split("_") if token]
    grades = [token for token in tokens if token in SEVERITY_NAMES]
    scorers = [token for token in tokens if token in SCORER_NAMES]
    unknown = [
        token
        for token in tokens
        if token not in SEVERITY_NAMES
        and token not in SCORER_NAMES
        and token not in _POOLED_TOKENS
    ]

    scorer = scorers[0] if len(scorers) == 1 else POOLED
    if unknown or len(grades) > 1:
        return text, scorer
    return (grades[0] if grades else POOLED), scorer


def with_severity(paired: pd.DataFrame, *, query_column: str = "query") -> pd.DataFrame:
    """`paired` plus the `severity` and `scorer` columns `split_query` derives.

    A copy, and a no-op for columns that are already there: `evaluate_bias` calls it on
    whatever it is handed, and a caller who resolved severity from the query *expressions*
    rather than the names should keep their version.
    """
    if set(BIAS_AXES) <= set(paired.columns):
        return paired
    if query_column not in paired.columns:
        raise KeyError(
            f"no {query_column!r} column to derive {BIAS_AXES} from; "
            f"available: {list(paired.columns)}"
        )

    derived = paired[query_column].map(split_query)
    frame = paired.copy()
    if "severity" not in frame.columns:
        frame["severity"] = [grade for grade, _ in derived]
    if "scorer" not in frame.columns:
        frame["scorer"] = [scorer for _, scorer in derived]
    return frame


def severity_order(values: Sequence[str]) -> list[str]:
    """`values` in ladder order: the pooled cell first, then the rungs, then the rest.

    The ladder is `STANDALONE_ORDER`. Unrecognised names (a pooled cell, a hand-added
    query) sort after the rungs rather than being guessed into the sequence.
    """
    seen = {str(value) for value in values}
    ladder = severity_scheme(sorted(seen)) or STANDALONE_ORDER

    known = [POOLED, *ladder]
    ordered = [grade for grade in known if grade in seen]
    return ordered + sorted(seen - set(ordered))


def scorer_order(values: Sequence[str]) -> list[str]:
    """`values` with the pooled cell first, then the rest alphabetically.

    The unfiltered view is the default read — `accept` is what the scorer *kept*, and it
    is only interesting against the whole set — so it leads wherever the axis is drawn.
    """
    seen = {str(value) for value in values}
    return ([POOLED] if POOLED in seen else []) + sorted(seen - {POOLED})


def quality_bias(
    real: ArrayLike, synth: ArrayLike, *, alpha: float = 0.05, min_n: int = 3
) -> dict[str, float | bool | None]:
    """Signed gap between a synthetic metric and its real counterpart, across models.

    `bias` is the mean of `synth - real`; positive means the synthetic images report the
    models as better than they are. `bias_sd` is the model-to-model spread of that gap and
    is the field that says whether the offset is a harmless rescaling or a ranking error —
    see the module docstring.

    Non-finite pairs are dropped pairwise and counted in `n_dropped`, as in
    `proxy_quality`: `aupro` is missing for whole backbones, and an all-NaN population
    must not read as a bias of zero. Below `min_n` models (or below 2, whichever binds)
    the spread fields are NaN — a single model's gap is a fact about that model, not
    about the synthetic set.
    """
    real = np.asarray(real, dtype=float).ravel()
    synth = np.asarray(synth, dtype=float).ravel()
    if real.shape != synth.shape:
        raise ValueError(f"real and synth differ in length: {real.size} vs {synth.size}")

    keep = np.isfinite(real) & np.isfinite(synth)
    n_dropped = int(keep.size - keep.sum())
    real, synth = real[keep], synth[keep]
    n = int(real.size)

    result: dict[str, float | bool | None] = {
        **dict.fromkeys(BIAS_FIELDS, _NAN),
        "n": n,
        "n_dropped": n_dropped,
        "alpha": alpha,
        "overestimates": None,
    }
    if n == 0:
        return result

    delta = synth - real
    result["bias"] = float(delta.mean())
    result["bias_median"] = float(np.median(delta))
    result["abs_bias"] = float(np.abs(delta).mean())
    result["max_abs_bias"] = float(np.abs(delta).max())
    result["over_share"] = float((delta > 0).mean())
    result["real_mean"] = float(real.mean())
    result["synth_mean"] = float(synth.mean())

    # Two models are the fewest that have a spread at all; `min_n` is the caller's
    # stricter floor on top of that, matching `correlate`.
    if n < max(min_n, 2):
        return result

    sd = float(delta.std(ddof=1))
    if sd <= _SPREAD_EPS * max(float(np.abs(delta).max()), 1.0):
        sd = 0.0
    se = sd / np.sqrt(n)
    result["bias_sd"], result["bias_se"] = sd, se

    half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
    lo, hi = float(result["bias"]) - half, float(result["bias"]) + half
    result["bias_ci_lo"], result["bias_ci_hi"] = lo, hi

    # `sd == 0` is the constant-offset population: the interval above has already
    # collapsed onto the estimate and answers `overestimates` on its own, so there is
    # nothing for the test to add and scipy is not asked to divide by zero.
    if sd > 0:
        test = stats.ttest_1samp(delta, 0.0)
        result["bias_t"] = float(test.statistic)  # type: ignore[union-attr]
        result["bias_p"] = float(test.pvalue)  # type: ignore[union-attr]

    if lo > 0:
        result["overestimates"] = True
    elif hi < 0:
        result["overestimates"] = False
    return result


def bias_keys(by: Sequence[str]) -> list[str]:
    """`by` with `query` replaced by the two axes it decomposes into.

    `query` is dropped rather than kept beside them because it is their product: keeping
    all three would leave the grouping unchanged and label it misleadingly. When the
    caller's `by` has no `query` at all the axes are appended anyway — averaging over
    severity is exactly what this statistic exists not to do.
    """
    keys: list[str] = []
    for key in by:
        if key == "query":
            keys.extend(axis for axis in BIAS_AXES if axis not in keys)
        elif key not in keys:
            keys.append(key)
    keys.extend(axis for axis in BIAS_AXES if axis not in keys)
    return keys


def evaluate_bias(
    paired: pd.DataFrame,
    by: Sequence[str] = DEFAULT_BIAS_BY,
    *,
    alpha: float = 0.05,
    min_n: int = 3,
    min_defect_samples: int | None = None,
    drop_proxy_only: bool = True,
) -> pd.DataFrame:
    """`quality_bias` per group of `by`, one row per group.

    `by` goes through `bias_keys`, so passing the same `--by` the rest of the analysis
    uses lands on the right axes without the caller restating them.

    `drop_proxy_only` removes the metrics whose two arms are not the same quantity
    (`PROXY_ONLY_METRICS`); leave it on unless you have a reason — see the module
    docstring. `min_defect_samples` screens thin queries exactly as `correlate` does.
    """
    keys = bias_keys(by)
    frame = with_severity(paired)

    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise KeyError(
            f"group keys {missing} not in paired frame; available: {list(frame.columns)}"
        )

    if drop_proxy_only and "metric" in frame.columns:
        frame = frame[~frame["metric"].isin(PROXY_ONLY_METRICS)]
    if min_defect_samples is not None and "n_defect_samples" in frame.columns:
        frame = frame[frame["n_defect_samples"] >= min_defect_samples]

    if frame.empty:
        return pd.DataFrame(columns=[*keys, *BIAS_FIELDS])

    rows = []
    # dropna=False for the same reason as `correlate`: a null group key is a real state
    # (older runs are missing some params) and the pandas default would delete it.
    for group_keys, group in frame.groupby(list(keys), dropna=False, sort=True):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        rows.append(
            dict(
                zip(keys, group_keys),
                **quality_bias(
                    group["real"], group["synth"], alpha=alpha, min_n=min_n
                ),
            )
        )

    result = pd.DataFrame.from_records(rows)
    return result.astype(
        {"n": "int64", "n_dropped": "int64", "overestimates": "boolean"}
    )


def worst_cells(bias: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    """The rows whose interval excludes zero, furthest from it first.

    The reporting view: a scan produces a few hundred cells and only the ones whose sign
    survived the spread are worth a line on screen.
    """
    if bias.empty or "overestimates" not in bias.columns:
        return bias
    decided = bias[bias["overestimates"].notna()]
    return decided.reindex(
        decided["bias"].abs().sort_values(ascending=False).index
    ).head(n)
