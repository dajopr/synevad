import pandas as pd
from omegaconf import DictConfig

# Logged as MLflow metrics on the `all` query. A subset of `ADMetrics`: the ordering
# statistics themselves, not their sample counts, which live in the artifact.
SEVERITY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "severity_cindex",
    "severity_cindex_defects",
    "severity_tau_b",
)


def summarize_metrics(
    frame: pd.DataFrame, config: DictConfig
) -> tuple[dict[str, float], dict[str, str]]:
    metrics = {}
    tags = {}

    no_filter = frame.query("query == 'all'")

    metrics["image_auroc"] = no_filter.image_auroc.item()
    metrics["pixel_auroc"] = no_filter.pixel_auroc.item()
    metrics["aupro"] = no_filter.aupro.item()

    # The ordinal statistics of the unfiltered set, so a run's severity behaviour is
    # visible in the MLflow UI without opening the metrics artifact. Guarded on presence:
    # runs written before `synevad.metrics.ordinal` landed have no such column, and NaN is
    # not a value MLflow accepts, so a real-arm run (every sample graded `real`) must skip
    # them rather than log a null.
    for column in SEVERITY_SUMMARY_COLUMNS:
        if column not in no_filter:
            continue
        value = no_filter[column].item()
        if pd.notna(value):
            metrics[column] = value

    filtered_any = frame.query("query != 'all'")
    if len(filtered_any) == 0:
        return metrics, tags

    metrics["n_good"] = no_filter.n_good_samples.item()
    metrics["n_defects_total"] = no_filter.n_defect_samples.item()
    metrics["n_defects_min"] = filtered_any.n_defect_samples.min()
    metrics["n_defects_max"] = filtered_any.n_defect_samples.max()

    has_na = filtered_any.isna().any(axis="index")
    too_few_defects = filtered_any.n_defect_samples < config.min_num_defects
    metrics["num_invalid"] = (has_na | too_few_defects).sum()

    metrics["positive_pixel_rate_mean"] = filtered_any.positive_pixel_rate.mean()

    tags["validity"] = "ok" if metrics["num_invalid"] == 0 else "error"
    return metrics, tags
