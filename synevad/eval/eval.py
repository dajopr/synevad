import tempfile
from pathlib import Path

import mlflow
from omegaconf import DictConfig

from synevad.db import log_params_and_tags, write_metrics, write_predictions
from synevad.eval.features import CachedBatch, score_cached_batch
from synevad.metrics import (
    build_predictions,
    compute_metrics_from_queries,
    queries_for,
    summarize_metrics,
)


def evaluate(model, cached: list[CachedBatch], set_name: str, config: DictConfig):
    """Score one eval set from features that were extracted once for the whole group."""
    batches = []
    run_name = f"{config.experiment_name}-{config.category}-{set_name}-{config.version}"

    queries = queries_for(config, set_name)
    with mlflow.start_run(run_name=run_name) as run:
        log_params_and_tags(set_name, config)

        for entry in cached:
            score, anomaly_map = score_cached_batch(model, entry)
            batches.append(entry.batch.with_predictions(score, anomaly_map))

        predictions = build_predictions(batches)
        metrics = compute_metrics_from_queries(predictions, queries)

        with tempfile.TemporaryDirectory() as tmp:
            write_predictions(
                predictions,
                Path(tmp),
                anomaly_maps=bool(config.logging.get("log_anomaly_maps", False)),
            )
            mlflow.log_artifacts(tmp, artifact_path="predictions")
        with tempfile.TemporaryDirectory() as tmp:
            write_metrics(metrics, Path(tmp), run.info.run_id)
            mlflow.log_artifacts(tmp, artifact_path="metrics")

        summary, tags = summarize_metrics(metrics, config)
        mlflow.log_metrics(summary)
        mlflow.set_tags(tags)

    return metrics
