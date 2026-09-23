"""Rebuild the ordinal severity metrics from artifacts a finished sweep already logged.

`synevad.metrics.ordinal` reads nothing new: the per-image `predictions/scores.csv` carries
`label`, `score` and `severity`, which is the whole input. So a sweep that ran before the
metric existed does not need re-running — `recompute_severity` rebuilds the columns from
what is on disk, and `augment_metrics` glues them onto the stored metrics frame.

What this cannot do is recompute the *pixel* metrics: `anomaly_maps.npz` is off by default
(348 MB per run), so `pixel_auroc`, `pixel_aupr` and `aupro` are not derivable from the
artifacts. They are therefore carried over from the stored parquet, and the result is that
frame **augmented** with the new columns — never a replacement for it.

`check_image_auroc` is the guard on all of it. The severity statistics are computed over
the same selected rows as the image metrics, so if the recomputed `image_auroc` reproduces
the stored one, the row selection did too. A disagreement means `scores.csv` and the
metrics parquet were written from different row sets, and the new columns cannot be trusted
*beside the stored ones* — they are internally consistent, just describing a different
population.

That is a real state and not a hypothetical: a query whose expression changed after the
sweep, or a gate naming a column `scores.csv` did not carry at eval time (`gate_queries`
drops it with a note, and a backfill over the stored runs adds such columns after the
fact), leaves the two eras apart. `augment_metrics(..., prefer_recomputed=True)` is the way
out — take the image metrics from the same rows as everything else recomputed — and the
check then measures how far the corpus moved rather than whether to trust it.

`exclude` drops severity grades from the ordinal statistics only — the recomputed
`image_auroc` the check rests on is always taken over the query's full row set, so a
filtered analysis is still checked against the stored value it must reproduce.

Feed the augmented frame to `build_paired_metrics(runs, metrics=...)`; from there the
severity columns are ordinary `metric` rows, paired against the real arm's `image_auroc`
by `PROXY_ONLY_METRICS`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from synevad.metrics import (
    SEVERITY_FIELDS,
    normalise_exclusions,
    queries_for,
    select_rows,
)
from synevad.metrics.ordinal import severity_concordance

# `run_id` and `query` together identify one scored row set, in both frames.
JOIN_KEYS: tuple[str, ...] = ("run_id", "query")

# The counts come along with the statistics: `n_severity_levels == 0` is how a row says
# the metric never applied (a real-arm run, or a query with one grade), which a NaN in the
# statistic alone cannot distinguish from a degenerate score vector. `SEVERITY_FIELDS`
# already holds both, so it is reused rather than restated.
SEVERITY_COLUMNS: tuple[str, ...] = tuple(SEVERITY_FIELDS)

CHECK_COLUMNS: tuple[str, ...] = (*JOIN_KEYS, "stored", "recomputed", "delta")


def recompute_run(
    scores: pd.DataFrame,
    queries: Sequence,
    *,
    min_level_n: int,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per query: the ordinal statistics, plus the image check.

    Row selection goes through `synevad.metrics.select_rows`, the same function
    `Predictions.select` uses, so a query means here exactly what it meant in the sweep.

    `exclude` reaches `severity_concordance` alone. The image metrics beside it stay
    unfiltered on purpose: they exist to reproduce the stored `image_auroc`, and dropping
    a grade from them would break the very check that says the severity columns can be
    trusted.
    """
    rows = []
    for query in queries:
        selected = scores.iloc[select_rows(scores, query.expr, query.keep_good)]
        label = selected["label"].to_numpy()
        score = selected["score"].to_numpy()

        # A query matching nothing, or only one class, scores to NaN rather than raising:
        # `min_num_defects` already screens these out downstream, and one bad query must
        # not cost the run its other eight.
        binary = np.unique(label).size > 1
        rows.append(
            {
                "query": query.name,
                "image_auroc_recomputed": (
                    float(roc_auc_score(label, score)) if binary else float("nan")
                ),
                "image_aupr_recomputed": (
                    float(average_precision_score(label, score))
                    if binary
                    else float("nan")
                ),
                **severity_concordance(
                    label,
                    score,
                    selected["severity"].astype(str).tolist(),
                    min_level_n=min_level_n,
                    exclude=exclude,
                ),
            }
        )
    return pd.DataFrame(rows)


def recompute_severity(
    runs: pd.DataFrame,
    config,
    *,
    min_level_n: int,
    exclude: Sequence[str] | None = None,
    read_scores=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """`recompute_run` over every run of `runs`, keyed by `run_id`.

    A run whose `tags.set_name` has no block under `config.queries` is skipped with a note
    rather than a `KeyError`: a store accumulates set names across sweeps, and one stale
    name must not stop the scan. Same for a run whose artifact will not read.

    `read_scores` is the `(run_id, artifact_uri) -> frame` reader, defaulting to
    `synevad.db.read_scores_artifact`; injectable so this is testable without a store.

    `exclude` is validated once here rather than per run: a bad grade name should fail on
    the first line of the scan, not after reading a few hundred artifacts.
    """
    if read_scores is None:
        from synevad.db import read_scores_artifact as read_scores

    exclude = sorted(normalise_exclusions(exclude))
    known = set(config.queries.keys())
    frames, skipped, failed = [], [], []

    for _, run in runs.iterrows():
        set_name = run.get("tags.set_name")
        if set_name not in known:
            skipped.append((run["run_id"], set_name))
            continue
        try:
            scores = read_scores(run["run_id"], run["artifact_uri"])
        except Exception as error:  # noqa: BLE001 — one unreadable run, not the scan
            failed.append((run["run_id"], type(error).__name__))
            continue

        per_query = recompute_run(
            scores,
            queries_for(config, set_name, scores),
            min_level_n=min_level_n,
            exclude=exclude,
        )
        per_query.insert(0, "run_id", run["run_id"])
        frames.append(per_query)

    if verbose:
        for label, entries in (("no query block", skipped), ("unreadable", failed)):
            if entries:
                head = ", ".join(f"{rid}({why})" for rid, why in entries[:3])
                print(
                    f"skipped {len(entries)} run(s), {label}: {head}"
                    f"{'...' if len(entries) > 3 else ''}",
                    file=sys.stderr,
                )

    if not frames:
        return pd.DataFrame(columns=["run_id", "query", *SEVERITY_COLUMNS])
    return pd.concat(frames, ignore_index=True)


# The image metrics `recompute_run` rebuilds from `scores.csv`, and the stored columns they
# stand in for. Both are computed from the same rows as the severity columns beside them,
# which is the whole point of `prefer_recomputed`.
RECOMPUTED_IMAGE_METRICS: dict[str, str] = {
    "image_auroc": "image_auroc_recomputed",
    "image_aupr": "image_aupr_recomputed",
}


def augment_metrics(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    prefer_recomputed: bool = False,
) -> pd.DataFrame:
    """The stored metrics frame plus the new columns, joined on (run_id, query).

    A left join off `stored`: the pixel metrics are only in the parquet and must survive,
    and a run present in one frame but not the other is a symptom worth seeing as NaN
    rather than silently dropping.

    `prefer_recomputed` replaces the stored `image_auroc` and `image_aupr` with the ones
    rebuilt here. Use it whenever a run's `scores.csv` no longer selects the rows the run
    scored — which is not exotic: a query whose expression changed, or a gate naming a
    column the file did not carry at eval time (a backfill over the stored runs adds
    those after the fact), leaves the logged image metrics describing one population and
    everything recomputed describing another. Mixing the two puts a selection rule reading
    the stored column and one reading the rebuilt column in the same table, compared as if
    they had seen the same images. `check_image_auroc` is how far apart they are.

    Off by default, because when the two *do* agree the stored value is the one the run
    published and there is no reason to restate it. The pixel metrics have no recomputed
    form — `anomaly_maps.npz` is not logged — so on a corpus where this matters they stay
    stale, and a selector reading a pixel column on the synthetic arm stays incomparable
    with one reading an image column. Report metrics come off the real arm, which is
    unaffected wherever its queries are ungated.
    """
    if stored.empty:
        return recomputed
    replaced = (
        {
            column: source
            for column, source in RECOMPUTED_IMAGE_METRICS.items()
            if prefer_recomputed and source in recomputed.columns
        }
        if prefer_recomputed
        else {}
    )
    new = recomputed[[*JOIN_KEYS, *SEVERITY_COLUMNS, *replaced.values()]]
    keep = [
        col
        for col in stored.columns
        if col not in SEVERITY_COLUMNS and col not in replaced
    ]
    merged = stored[keep].merge(new, on=list(JOIN_KEYS), how="left")
    return merged.rename(
        columns={source: column for column, source in replaced.items()}
    )


def check_image_auroc(stored: pd.DataFrame, recomputed: pd.DataFrame) -> pd.DataFrame:
    """Recomputed vs. stored `image_auroc`, per (run, query), worst disagreement first.

    See the module docstring: this is the guard the severity columns rest on.
    """
    if stored.empty or "image_auroc" not in stored.columns:
        return pd.DataFrame(columns=list(CHECK_COLUMNS))

    merged = stored[[*JOIN_KEYS, "image_auroc"]].merge(
        recomputed[[*JOIN_KEYS, "image_auroc_recomputed"]],
        on=list(JOIN_KEYS),
        how="inner",
    )
    merged = merged.rename(
        columns={"image_auroc": "stored", "image_auroc_recomputed": "recomputed"}
    )
    merged["delta"] = (merged["stored"] - merged["recomputed"]).abs()
    return merged.sort_values("delta", ascending=False, ignore_index=True)
