from pathlib import Path

import numpy as np
import pandas as pd

from synevad.metrics import Predictions


def write_metrics(metrics: pd.DataFrame, root: Path, run_id: str) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    metrics.to_parquet(root / f"{run_id}.parquet")


def write_predictions(
    predictions: Predictions, root: Path, *, anomaly_maps: bool = False
) -> None:
    """Writes per-image predictions out for `mlflow.log_artifacts`.

    Under `root`: `scores.csv` (one row per image), and with `anomaly_maps=True` also
    `anomaly_maps.npz` (raw maps, keyed by sample id).

    The maps are off by default. Every metric the sweep logs is computed from them
    before they are written, so the artifact only saves a re-run for analysis that wants
    the pixel maps themselves (`eda.ipynb`) — and at 348 MB per run that trade stopped
    being worth it once scoring a set became cheap.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    predictions.frame.to_csv(root / "scores.csv", index=False)
    if not anomaly_maps:
        return

    maps = dict(zip(predictions.frame["sample_id"], predictions.anomaly_map))
    # Uncompressed on purpose: float32 anomaly maps are near-incompressible, so
    # `savez_compressed` spent 8.8s to save 13% of a 348 MB payload at 384px. `savez`
    # writes the same arrays losslessly in 0.14s.
    np.savez(root / "anomaly_maps.npz", **maps)
