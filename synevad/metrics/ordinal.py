"""Ordinal severity statistics: does the score track *how bad* the defect is.

`image_auroc` asks one question — can the model separate defective images from clean
ones. The synthetic corpus can answer a strictly harder one, because every generated
defect carries the severity grade it was asked for, and the real test set has no such
label. Whether a model orders `minimal < slight < moderate < severe` correctly is
information the real arm cannot supply and the binary metric cannot see.

The statistic is the **C-index**: over every pair of images from *different* severity
levels, the fraction where the more severe one scored higher (ties counted a half). With
two levels it reduces exactly to the tie-corrected `image_auroc`, so it is a strict
generalisation rather than a different scale, and it is computed as a
count-weighted mean of the per-level-pair AUCs

    c_index = sum_{a<b} n_a n_b AUC(a, b) / sum_{a<b} n_a n_b

which costs one `rankdata` per level pair rather than an O(n^2) pass, and yields the
`level_pair_auc` matrix as a by-product — the diagnostic that says *which* step of the
ladder a model cannot resolve.

Two readings are reported and they are not interchangeable:

* `cindex` includes the defect-free images as level 0, so it contains the whole of
  `image_auroc` plus the severity ordering.
* `cindex_defects` drops them, so it is *only* the severity ordering. This is the number
  that answers "does the ordinal proxy know anything AUROC does not" — a model can be
  perfect at `cindex` purely by separating good from bad.

`tau_b` is the stricter reading (it penalises score spread *within* a level, where the
C-index does not care) and `spearman` the rank correlation over defects alone.

`exclude` drops whole severity levels before anything is computed, without renumbering
the rest: the ladder positions come from the vocabulary, not from what survived, so
`severity_cindex` with `slight` excluded is the same statistic read over one rung fewer,
directly comparable to the unfiltered one. It is the knob for the two questions the
corpus keeps raising — whether the ladder still orders once the grade the generator is
worst at is taken out, and (with `none` excluded) what the ordering alone is worth.

Nothing here warns. Degenerate inputs — one severity level, an unrecognised vocabulary,
an all-`real` frame — return NaN fields, matching `synevad.analysis.correlation`, so a
scan over a few thousand runs stays readable. An unknown *exclusion* is the exception and
raises: it is a caller's typo, and silently filtering nothing would report the unfiltered
statistic under a filtered name.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

# Defect-free first, then the four standalone grades. See `synevad.data.synevad.Severity`.
STANDALONE_ORDER: tuple[str, ...] = ("none", "minimal", "slight", "moderate", "severe")
LADDERS: tuple[tuple[str, ...], ...] = (STANDALONE_ORDER,)
SEVERITY_NAMES: frozenset[str] = frozenset(STANDALONE_ORDER)

# The dataclass default carried by every sample of a real dataset (`synevad.data.Sample`).
# Real defects have no grade, so a real-arm frame ranks to nothing and reports NaN — which
# is the correct answer, not a failure: this metric exists only on the synthetic arm.
UNGRADED: frozenset[str] = frozenset({"real", "", "nan"})

# Fixed key set, so `pd.DataFrame.from_records` over a scan gets a stable schema no matter
# which branch each group took.
SEVERITY_FIELDS: tuple[str, ...] = (
    "severity_cindex",
    "severity_cindex_defects",
    "severity_tau_b",
    "severity_spearman",
    "severity_adjacent_auc",
    "n_severity_levels",
    "n_severity_pairs",
)

_NAN = float("nan")


def empty_severity() -> dict[str, float]:
    """The all-NaN result, for callers with no `severity` column to read.

    Counts are 0 rather than NaN: "no levels were found" is a fact, not a missing value,
    and `n_severity_levels == 0` is how a frame says the metric never applied.
    """
    return {
        **dict.fromkeys(SEVERITY_FIELDS, _NAN),
        "n_severity_levels": 0.0,
        "n_severity_pairs": 0.0,
    }


def severity_scheme(values: Sequence[str]) -> tuple[str, ...] | None:
    """`STANDALONE_ORDER` if `values` uses those grades, else None.

    None for an all-`real` frame from the real arm, or a frame whose only recognised
    grade is `none` (defect-free only). Unknown grades are ignored here and NaN-ranked
    in `severity_ranks`.
    """
    names = {str(value).strip().lower() for value in values}
    names = (names - UNGRADED) & SEVERITY_NAMES
    if not names - {"none"}:
        return None
    return STANDALONE_ORDER


def severity_ranks(
    values: Sequence[str], *, order: Sequence[str] | None = None
) -> np.ndarray:
    """Integer ranks for `values` along `order`, NaN for anything not in it.

    NaN rather than an exception for the unknown ones: a corpus carrying a stray grade
    should cost those rows, not the whole statistic. `order` defaults to the scheme
    `severity_scheme` detects.
    """
    order = order if order is not None else severity_scheme(values)
    if order is None:
        return np.full(len(values), _NAN)

    lookup = {name: rank for rank, name in enumerate(order)}
    return np.array(
        [lookup.get(str(value).strip().lower(), _NAN) for value in values], dtype=float
    )


def normalise_exclusions(names: Sequence[str] | None) -> frozenset[str]:
    """Lowercased, whitespace-stripped `names`, refusing a grade the ladder does not spell.

    The one place an exclusion is validated, so the CLI and the metric reject the same
    typos with the same message. Raising is deliberate — see the module docstring: a
    misspelled `--exclude-severity sever` that quietly filtered nothing would publish the
    unfiltered statistic under a filtered name, which is worse than a stack trace.
    """
    if not names:
        return frozenset()

    cleaned = frozenset(str(name).strip().lower() for name in names)
    unknown = cleaned - SEVERITY_NAMES
    if unknown:
        raise ValueError(
            f"unknown severity grade(s) {sorted(unknown)}; "
            f"the ladder is {list(STANDALONE_ORDER)}"
        )
    return cleaned


def _excluded_mask(values: Sequence[str], excluded: frozenset[str]) -> np.ndarray:
    """Boolean mask of the rows whose grade is excluded, aligned with `values`."""
    return np.array(
        [str(value).strip().lower() in excluded for value in values], dtype=bool
    )


def _auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """P(positive > negative) + 0.5 P(tie), via the tie-corrected Mann-Whitney statistic.

    Identical to `sklearn.metrics.roc_auc_score` on the two-class problem, but takes the
    two score vectors directly, which is what the level-pair loop has in hand.
    """
    n_neg, n_pos = negative.size, positive.size
    if n_neg == 0 or n_pos == 0:
        return _NAN
    ranks = stats.rankdata(np.concatenate([negative, positive]))
    rank_sum = float(ranks[n_neg:].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_neg * n_pos)


def level_pair_auc(
    ranks: np.ndarray, score: np.ndarray, levels: Sequence[int]
) -> dict[tuple[int, int], float]:
    """AUC of every ordered level pair `(a, b)`, `a < b`, as {(a, b): AUC}.

    The diagnostic behind the summary numbers: a model can hold a respectable C-index
    while being blind to one rung of the ladder, and only the pair matrix shows it. The
    `(0, 1)` entry — clean against the mildest grade — is normally the hard one.
    """
    by_level = {level: score[ranks == level] for level in levels}
    return {
        (a, b): _auc(by_level[a], by_level[b])
        for index, a in enumerate(levels)
        for b in levels[index + 1 :]
    }


def _weighted_cindex(
    pairs: dict[tuple[int, int], float], counts: dict[int, int]
) -> tuple[float, float]:
    """Count-weighted mean of `pairs`, and the number of comparable pairs behind it.

    Weighting by `n_a * n_b` rather than treating each level pair equally is what makes
    the two-level case reduce to `image_auroc` exactly, and it keeps a grade the
    generator happened to produce three of from carrying the same weight as one with two
    hundred.
    """
    weights = {(a, b): counts[a] * counts[b] for a, b in pairs}
    total = float(sum(weights.values()))
    if total <= 0:
        return _NAN, 0.0
    weighted = sum(pairs[key] * weights[key] for key in pairs)
    return float(weighted / total), total


def severity_concordance(
    label: ArrayLike,
    score: ArrayLike,
    severity: Sequence[str],
    *,
    order: Sequence[str] | None = None,
    min_level_n: int = 1,
    exclude: Sequence[str] | None = None,
) -> dict[str, float]:
    """C-index and friends for one scored set. NaN fields when the set cannot support them.

    `label` is only used to identify the defect-free rows for `severity_cindex_defects`;
    the ordering itself comes from `severity`, whose defect-free grade is `none`.

    `min_level_n` drops severity levels with too few images before anything is computed —
    a level of one contributes a pair count but no information, and on a per-query frame
    (`severity == 'severe'` plus the good images) that is a common shape. Levels dropped
    this way are excluded from `n_severity_levels` as well, so the field reads as "levels
    the statistic actually rests on".

    `exclude` names severity grades to drop entirely — their images take no part in any
    of the statistics, and the levels disappear from `n_severity_levels` and
    `n_severity_pairs` exactly as `min_level_n` drops a thin one. The surviving levels
    keep their ladder positions, so excluding `slight` leaves `minimal` and `moderate`
    two rungs apart rather than adjacent, and the result stays on the same scale as the
    unfiltered one. Two uses: excluding a middle grade asks whether the ordering survives
    without the rung the generator is worst at, and excluding `none` drops the defect-free
    images so `severity_cindex` becomes the ordering alone. Detection of the vocabulary
    happens *before* the exclusion, so filtering the only grade that identifies a ladder
    does not make the rest unrankable.

    On a single-grade query the result is `severity_cindex == image_auroc` by
    construction, and `severity_cindex_defects` is NaN — there is only one defect level to
    order. The metric is meant for the `all` query, or any query spanning more than one
    grade.
    """
    label = np.asarray(label).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if not (label.size == score.size == len(severity)):
        raise ValueError(
            f"label, score and severity differ in length: "
            f"{label.size} vs {score.size} vs {len(severity)}"
        )

    result = empty_severity()
    excluded = normalise_exclusions(exclude)

    ranks = severity_ranks(severity, order=order)
    if excluded and ranks.size:
        # After ranking, so the ladder is still detected from the full vocabulary, and the
        # kept grades hold the positions they have in it.
        ranks = np.where(_excluded_mask(severity, excluded), _NAN, ranks)
    keep = np.isfinite(ranks) & np.isfinite(score)
    ranks, score, label = ranks[keep], score[keep], label[keep]
    if ranks.size == 0:
        return result

    present, counts = np.unique(ranks, return_counts=True)
    levels = [int(level) for level, n in zip(present, counts) if n >= min_level_n]
    count_of = {int(level): int(n) for level, n in zip(present, counts)}
    result["n_severity_levels"] = float(len(levels))
    if len(levels) < 2:
        return result

    in_levels = np.isin(ranks, levels)
    ranks, score, label = ranks[in_levels], score[in_levels], label[in_levels]

    pairs = level_pair_auc(ranks, score, levels)
    cindex, n_pairs = _weighted_cindex(pairs, count_of)
    result["severity_cindex"] = cindex
    result["n_severity_pairs"] = n_pairs

    # Adjacent rungs only: the mean over consecutive *present* levels, so a corpus missing
    # a grade compares the two levels that are adjacent in what it has, not in the ladder.
    adjacent = [pairs[(a, b)] for a, b in zip(levels, levels[1:])]
    if adjacent:
        result["severity_adjacent_auc"] = float(np.mean(adjacent))

    # Defects only: strip level 0 *and* anything the caller labelled good, so a query with
    # `keep_good=False` and a mislabelled row cannot smuggle the binary signal back in.
    defect_levels = [level for level in levels if level > 0]
    if len(defect_levels) >= 2:
        defect_pairs = {
            key: value
            for key, value in pairs.items()
            if key[0] in defect_levels and key[1] in defect_levels
        }
        result["severity_cindex_defects"] = _weighted_cindex(defect_pairs, count_of)[0]

        # Both arms need variance or scipy warns and returns NaN; a constant score
        # across defects is a real outcome (a model that saturates), not an error.
        is_defect = (ranks > 0) & (np.asarray(label) != 0)
        if (
            is_defect.sum() >= 3
            and np.unique(ranks[is_defect]).size >= 2
            and np.unique(score[is_defect]).size >= 2
        ):
            spearman = stats.spearmanr(ranks[is_defect], score[is_defect])
            result["severity_spearman"] = float(spearman.statistic)  # type: ignore

    # tau-b over every kept row, the strict reading: unlike the C-index it charges for
    # score spread inside a level, which is why it sits below `severity_cindex` in
    # practice rather than tracking it.
    if np.unique(score).size > 1:
        result["severity_tau_b"] = float(stats.kendalltau(ranks, score).statistic)  # type: ignore

    return result


def severity_report(
    frame: pd.DataFrame,
    *,
    by: Sequence[str] = ("query",),
    order: Sequence[str] | None = None,
    min_level_n: int = 1,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    """`severity_concordance` per group of `by` over a `scores.csv`-shaped frame.

    Expects the `label`, `score` and `severity` columns `build_predictions` writes. Used
    by `synevad.analysis.severity` to recompute the statistic from artifacts already on
    disk, without re-running a sweep.
    """
    required = {"label", "score", "severity"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing {sorted(missing)}; has {list(frame.columns)}")

    by = [key for key in by if key in frame.columns]
    if not by:
        return pd.DataFrame(
            [
                severity_concordance(
                    frame["label"],
                    frame["score"],
                    frame["severity"].tolist(),
                    order=order,
                    min_level_n=min_level_n,
                    exclude=exclude,
                )
            ]
        )

    rows = []
    for keys, group in frame.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            dict(
                zip(by, keys),
                **severity_concordance(
                    group["label"],
                    group["score"],
                    group["severity"].tolist(),
                    order=order,
                    min_level_n=min_level_n,
                    exclude=exclude,
                ),
            )
        )
    return pd.DataFrame.from_records(rows)
