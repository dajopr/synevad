"""Pairing real and synthetic runs into one frame ready for correlation.

`setup_and_evaluate` trains one PatchCore model per sweep point and then logs one MLflow
run per dataloader, so a real run and its synthetic sibling(s) differ only in
`tags.set_name`. Recovering that pairing from the flat `search_runs` frame is the whole
job of this module; the statistics live in `synevad.analysis.correlation`.
"""

import warnings
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from synevad.db.read import get_ad_metrics, get_metadata_by_version

# The `ADMetrics` fields worth correlating: the score columns, not the sample counts.
METRIC_COLUMNS: tuple[str, ...] = (
    "image_auroc",
    "pixel_auroc",
    "image_aupr",
    "pixel_aupr",
    "aupro",
    "severity_cindex",
    "severity_cindex_defects",
    "severity_tau_b",
    "severity_spearman",
    "severity_adjacent_auc",
)

# Metrics that exist on the synthetic arm alone, mapped to the real metric whose ranking
# each is meant to predict.
#
# The severity statistics (`synevad.metrics.ordinal`) need a per-image severity grade, and
# real defects have none — so there is no real-arm counterpart to pair them against, and
# the plain merge on `metric` would leave every one of these rows with `real = NaN` and
# drop the whole metric from the analysis without saying so.
#
# Broadcasting the real arm's `image_auroc` onto them is not a fudge, it is the question
# being asked: *does this synthetic-only statistic rank models the way the real image
# AUROC does?* Read a `severity_cindex` row of the paired frame as "synth = the ordinal
# concordance on the synthetic set, real = the real image AUROC of the same model", which
# is exactly what `selection_regret` then needs — regret is always denominated in the real
# metric, whatever the proxy was.
PROXY_ONLY_METRICS: dict[str, str] = {
    "severity_cindex": "image_auroc",
    "severity_cindex_defects": "image_auroc",
    "severity_tau_b": "image_auroc",
    "severity_spearman": "image_auroc",
    "severity_adjacent_auc": "image_auroc",
}

REAL_SET_NAME = "real"

# `config.queries.real` has one entry, so the real arm is scored on the full test set.
REAL_QUERY = "all"


def default_group_keys(runs: pd.DataFrame) -> list[str]:
    """Columns that identify one trained model, i.e. one sweep point.

    Every swept key becomes a `params.*` column (`apply_overrides` logs it twice, dotted
    and aliased), so the params tuple alone would usually do. Category and version are
    named explicitly anyway: they are the columns a reader expects to see partitioning
    the population, and the duplicate params are functionally dependent, so listing extra
    keys only widens the tuple, never splits a group.
    """
    params = sorted(col for col in runs.columns if col.startswith("params."))
    return [
        col for col in ("tags.category", "tags.version") if col in runs.columns
    ] + params


def _pick_one(
    group: pd.DataFrame, duplicates: Literal["latest", "error"], label: str
) -> pd.Series:
    """One run out of a group of re-runs of the same config."""
    if len(group) == 1:
        return group.iloc[0]

    if duplicates == "error":
        ids = ", ".join(group["run_id"])
        raise ValueError(f"{len(group)} {label} runs for one config: {ids}")

    # Newest wins. Deliberately not the cartesian product of duplicates: re-running a
    # config yields correlated points, which would inflate n and flatter every result.
    return group.loc[group["start_time"].idxmax()]


def pair_runs(
    runs: pd.DataFrame,
    *,
    group_keys: Sequence[str] | None = None,
    duplicates: Literal["latest", "error"] = "latest",
    on_unpaired: Literal["drop", "error"] = "drop",
) -> pd.DataFrame:
    """One row per (model config, synthetic set), linking a real run to its sibling.

    Columns: `model_id`, `real_run_id`, `synth_run_id`, `set_name`, `category`,
    `version`, and every `params.<k>` carried through as `param_<k>` (the rename avoids
    colliding `params.category` with the category column).

    Distinct `set_name`s are separate rows, not duplicates — that is how more than one
    entry under `config.data.synthetic` is handled with no special case.
    """
    keys = list(group_keys) if group_keys is not None else default_group_keys(runs)
    missing = [key for key in keys if key not in runs.columns]
    if missing:
        raise KeyError(f"group keys {missing} not in runs frame")
    if "tags.set_name" not in runs.columns:
        raise KeyError(
            "runs frame has no `tags.set_name`; cannot tell real from synthetic"
        )

    param_cols = [col for col in runs.columns if col.startswith("params.")]

    rows: list[dict] = []
    model_id = 0
    n_deduped = 0
    no_synth: list[str] = []
    no_real: list[str] = []

    # dropna=False: `params.seed` and friends are null for older runs, and the pandas
    # default would silently delete those sweep points instead of grouping them.
    for _, group in runs.groupby(keys, dropna=False, sort=True):
        real = group[group["tags.set_name"] == REAL_SET_NAME]
        synth = group[group["tags.set_name"] != REAL_SET_NAME]

        if real.empty or synth.empty:
            (no_real if real.empty else no_synth).append(str(group["run_id"].iloc[0]))
            continue

        n_deduped += int(len(real) > 1)
        real_run = _pick_one(real, duplicates, "real")

        for set_name, per_set in synth.groupby(
            "tags.set_name", dropna=False, sort=True
        ):
            n_deduped += int(len(per_set) > 1)
            synth_run = _pick_one(per_set, duplicates, f"synthetic ({set_name})")
            rows.append(
                {
                    "model_id": model_id,
                    "real_run_id": real_run["run_id"],
                    "synth_run_id": synth_run["run_id"],
                    "set_name": set_name,
                    "category": real_run.get("tags.category"),
                    "version": real_run.get("tags.version"),
                    **{
                        f"param_{col.removeprefix('params.')}": real_run[col]
                        for col in param_cols
                    },
                }
            )
        model_id += 1

    unpaired = no_real + no_synth
    if unpaired:
        message = (
            f"dropped {len(unpaired)} unpaired config(s): "
            f"{len(no_real)} without a real run, {len(no_synth)} without a synthetic one"
        )
        if on_unpaired == "error":
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    if n_deduped:
        warnings.warn(
            f"kept the latest of re-run configs in {n_deduped} group(s)", stacklevel=2
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_id",
                "real_run_id",
                "synth_run_id",
                "set_name",
                "category",
                "version",
                *(f"param_{col.removeprefix('params.')}" for col in param_cols),
            ]
        )
    return pd.DataFrame(rows)


def _melt_metrics(
    metrics: pd.DataFrame, value_name: str, keep: Sequence[str]
) -> pd.DataFrame:
    return metrics.melt(
        id_vars=["run_id", *keep],
        value_vars=[col for col in METRIC_COLUMNS if col in metrics.columns],
        var_name="metric",
        value_name=value_name,
    )


def _with_proxy_targets(
    real_long: pd.DataFrame, mapping: dict[str, str] = PROXY_ONLY_METRICS
) -> pd.DataFrame:
    """Give every proxy-only metric the real value of the metric it stands in for.

    Any proxy-only rows the real arm produced on its own are dropped first: a real run
    that happens to carry a `severity_cindex` column carries it as NaN (its samples are
    all graded `real`), and keeping that row would shadow the broadcast one.
    """
    kept = real_long[~real_long["metric"].isin(mapping)]
    extra = [
        kept[kept["metric"] == target].assign(metric=proxy)
        for proxy, target in mapping.items()
        if (kept["metric"] == target).any()
    ]
    return pd.concat([kept, *extra], ignore_index=True) if extra else kept


def _real_scores(metrics: pd.DataFrame, real_ids: set[str]) -> pd.DataFrame:
    """The real arm's single scored row per run, melted to one row per metric.

    Proxy-only metrics are filled in from `PROXY_ONLY_METRICS` — see the note there.
    """
    real = metrics[metrics["run_id"].isin(real_ids)]

    extra = real.groupby("run_id")["query"].nunique().gt(1)
    if extra.any():
        warnings.warn(
            f"{int(extra.sum())} real run(s) have more than one query; "
            f"using query == {REAL_QUERY!r}",
            stacklevel=3,
        )
    real = real[real["query"] == REAL_QUERY]

    melted = _melt_metrics(real, "real", keep=())
    return _with_proxy_targets(melted).rename(columns={"run_id": "real_run_id"})


def build_paired_metrics(
    runs: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    *,
    group_keys: Sequence[str] | None = None,
    duplicates: Literal["latest", "error"] = "latest",
    on_unpaired: Literal["drop", "error"] = "drop",
) -> pd.DataFrame:
    """Long frame with one row per (model config, synthetic query, metric).

    Columns: `model_id`, `category`, `version`, `set_name`, `query`, `metric`, `real`,
    `synth`, `real_run_id`, `synth_run_id`, `n_samples`, `n_defect_samples`, `param_*`.

    The real arm is scored only on `query == 'all'`, so its value is **broadcast** across
    every synthetic query: each row asks "does the synthetic set, restricted to query Q,
    rank models the way the full real test set does?". `real` therefore repeats once per
    query within a model, and its variance is over models alone.

    The severity metrics are broadcast a second time, across the `metric` axis: they have
    no real-arm counterpart, so `real` on a `severity_cindex` row is the model's real
    `image_auroc`. See `PROXY_ONLY_METRICS`.

    Pass `metrics` to reshape a frame you already read; it defaults to reading every
    run's artifact. Note `query` shadows `DataFrame.query`, so index it as `frame["query"]`.
    """
    pairs = pair_runs(
        runs, group_keys=group_keys, duplicates=duplicates, on_unpaired=on_unpaired
    )
    if metrics is None:
        metrics = get_ad_metrics(runs)

    if pairs.empty or metrics.empty:
        return pd.DataFrame(
            columns=[
                *pairs.columns,
                "query",
                "metric",
                "real",
                "synth",
                "n_samples",
                "n_defect_samples",
            ]
        )

    synth_long = _melt_metrics(
        metrics[metrics["run_id"].isin(set(pairs["synth_run_id"]))],
        "synth",
        keep=("query", "n_samples", "n_defect_samples"),
    ).rename(columns={"run_id": "synth_run_id"})

    real_long = _real_scores(metrics, set(pairs["real_run_id"]))

    paired = pairs.merge(synth_long, on="synth_run_id", how="inner")
    paired = paired.merge(real_long, on=["real_run_id", "metric"], how="left")

    ordered = [
        "model_id",
        "category",
        "version",
        "set_name",
        "query",
        "metric",
        "real",
        "synth",
        "real_run_id",
        "synth_run_id",
        "n_samples",
        "n_defect_samples",
    ]
    rest = [col for col in paired.columns if col not in ordered]
    return paired[[*ordered, *rest]].reset_index(drop=True)


def load_paired_metrics(version: str, **kwargs) -> pd.DataFrame:
    """`build_paired_metrics` for every run of one sweep version.

    Assumes the tracking URI is already set (see `synevad.db.tracking_uri_from_env`).
    """
    return build_paired_metrics(get_metadata_by_version(version), **kwargs)
