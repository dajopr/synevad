"""Read side of the MLflow store: run metadata and the per-run metric artifacts.

Mirror of `synevad.db.store`, which writes `metrics/{run_id}.parquet` per run. Nothing
here knows about experiment semantics (real vs synthetic runs, pairing); that lives in
`synevad.analysis.data`.

Reading artifacts off a local store
-----------------------------------
Against a tracking server, artifacts carry `mlflow-artifacts:` URIs and every read is an
HTTP round trip to the server plus a copy into a temp dir — about 95 ms per file here,
which over a 26k-run sweep is 40 minutes of the analysis just fetching 18 KB CSVs. When
the analysis runs on the *same machine* as the server, those files are already on local
disk, and reading them in place costs 0.8 ms instead.

Set `MLFLOW_LOCAL_ARTIFACT_ROOT` to the server's `--artifacts-destination` directory (the
one holding `<experiment_id>/<run_id>/artifacts/...`) and `artifact_path` resolves
`mlflow-artifacts:` URIs against it, falling back to the download whenever the file is
not there. Both readers go through it, so the severity recompute and the certainty
analysis get the shortcut alike. Leaving it unset keeps the old behaviour exactly. The fallback is what makes
this safe to leave on: a wrong or stale root costs a `stat` per file and then downloads,
it cannot serve the wrong run, because the run id is part of the path.
"""

import os
import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

# Imported for `tracking_uri_from_config`, but the import is also what loads `.env`, so
# the `${oc.env:...}` interpolations in `mvtec.yaml` below resolve.
from synevad.db.logging import tracking_uri_from_config

# Column schema of `metrics/{run_id}.parquet`: `query` plus the fields of
# `synevad.metrics.ADMetrics`, in the order `ADMetricsCollection.as_frame` emits them.
AD_METRIC_COLUMNS: tuple[str, ...] = (
    "query",
    "image_auroc",
    "pixel_auroc",
    "image_aupr",
    "pixel_aupr",
    "aupro",
    "n_samples",
    "n_good_samples",
    "n_defect_samples",
    "positive_pixel_rate",
    "severity_cindex",
    "severity_cindex_defects",
    "severity_tau_b",
    "severity_spearman",
    "severity_adjacent_auc",
    "n_severity_levels",
    "n_severity_pairs",
)

# Column schema of `predictions/scores.csv`, in the order `build_predictions` emits them.
# One row per scored image, which is what makes a per-image analysis possible at all.
SCORE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "class_name",
    "label",
    "mask_path",
    "score",
    "scorer",
    "severity",
    "mask_status",
    "reward_score",
    "defect_area_frac",
)

# MLflow's own pandas cap (`SEARCH_MAX_RESULTS_PANDAS`). Restated rather than imported so
# a version bump cannot move it under the truncation warning below.
MAX_RUNS = 100_000


# The URI scheme a tracking server hands out when it serves artifacts itself. The path
# after it is `<experiment_id>/<run_id>/artifacts/...`, relative to the server's
# `--artifacts-destination`, which is what `MLFLOW_LOCAL_ARTIFACT_ROOT` names.
ARTIFACT_SCHEME = "mlflow-artifacts:"

LOCAL_ROOT_ENV = "MLFLOW_LOCAL_ARTIFACT_ROOT"


@lru_cache(maxsize=1)
def local_artifact_root() -> Path | None:
    """`MLFLOW_LOCAL_ARTIFACT_ROOT` as a directory, or None if unset or not one.

    Cached because it is consulted once per artifact and a sweep has tens of thousands;
    call `local_artifact_root.cache_clear()` after changing the variable (tests do).

    A path that is not a directory warns rather than raising: the fallback download still
    works, so a stale `.env` on one machine should slow the analysis down, not stop it.
    """
    raw = os.environ.get(LOCAL_ROOT_ENV, "").strip()
    if not raw:
        return None

    root = Path(raw).expanduser()
    if not root.is_dir():
        warnings.warn(
            f"{LOCAL_ROOT_ENV}={raw!r} is not a directory; falling back to downloading "
            "artifacts through the tracking server",
            stacklevel=2,
        )
        return None
    return root


def local_artifact_path(artifact_uri: str, relative: str) -> Path | None:
    """Where `<artifact_uri>/<relative>` would sit on this filesystem, or None.

    Two URI forms reach here. A `file://` location is already a local path, so it is
    parsed directly — `download_artifacts` would return the same path, but only after
    building an artifact repository to do it. An `mlflow-artifacts:` location is resolved
    against `local_artifact_root`, which is the only case that needs configuring.

    None means "no idea", not "missing": the caller downloads. Existence is checked by
    `artifact_path`, not here, so this stays a pure path computation.
    """
    if artifact_uri.startswith("file://"):
        return Path(url2pathname(urlparse(artifact_uri).path)) / relative

    root = local_artifact_root()
    if root is None or not artifact_uri.startswith(ARTIFACT_SCHEME):
        return None
    return root / artifact_uri[len(ARTIFACT_SCHEME) :].lstrip("/") / relative


def artifact_path(artifact_uri: str, relative: str) -> str:
    """Filesystem path for `<artifact_uri>/<relative>`, downloading only if it must.

    The one place the local-store shortcut is taken, so both artifact readers get it and
    neither has to know the scheme rules. See the module docstring for why it matters and
    why the fallback makes it safe.
    """
    local = local_artifact_path(artifact_uri, relative)
    if local is not None and local.is_file():
        return str(local)
    return mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/{relative}"
    )


def tracking_uri_from_env() -> str:
    """The tracking URI `setup_mlflow` would build, without its write side effects.

    Read paths must not `mkdir` or create experiments, so callers do
    `mlflow.set_tracking_uri(tracking_uri_from_env())` themselves. Both sides share
    `tracking_uri_from_config` so `MLFLOW_TRACKING_URI` and the SQLite fallback stay
    in one place.
    """
    config = cast(
        DictConfig, OmegaConf.load(Path(__file__).parent.parent / "config" / "mvtec.yaml")
    )
    return tracking_uri_from_config(config)


def get_experiment_runs(
    experiment_names: Sequence[str],
    *,
    filter_string: str = "",
    finished_only: bool = True,
    max_results: int = MAX_RUNS,
) -> pd.DataFrame:
    """One row per matching run, with the `run_id`/`metrics.*`/`params.*`/`tags.*` columns
    `mlflow.search_runs` returns.

    `max_results` is passed explicitly because `search_runs` truncates at
    `SEARCH_MAX_RESULTS_PANDAS` without saying so, and a store that has accumulated a few
    sweeps gets there. Truncation is safe for every caller — a missing run only ever means
    more work, never a wrong answer — but it is invisible, hence the warning.
    """
    metadata = mlflow.search_runs(
        experiment_names=list(experiment_names),
        filter_string=filter_string,
        max_results=max_results,
    )
    assert isinstance(metadata, pd.DataFrame)

    if len(metadata) == max_results:
        warnings.warn(
            f"the tracking store returned the maximum of {max_results} runs for "
            f"{list(experiment_names)}; older runs are missing from this frame",
            stacklevel=2,
        )

    # Unfinished runs never got as far as logging the metrics artifact.
    if finished_only and len(metadata):
        metadata = metadata[metadata["status"] == "FINISHED"].reset_index(drop=True)
    return metadata


def get_metadata_by_version(
    version: str,
    experiment_names: Sequence[str] = ("mvtec",),
    *,
    finished_only: bool = True,
) -> pd.DataFrame:
    """One row per run of `version`, with the `run_id`/`metrics.*`/`params.*`/`tags.*`
    columns `mlflow.search_runs` returns."""
    return get_experiment_runs(
        experiment_names,
        filter_string=f"tags.version='{version}'",
        finished_only=finished_only,
    )


def read_metrics_artifact(run_id: str, artifact_uri: str) -> pd.DataFrame:
    """The per-query metrics frame written by `write_metrics`, with `run_id` prepended.

    Resolves through `artifact_uri` rather than `run_id`: the `run_id` form goes via
    `RunsArtifactRepository`, which resolves the run's location on the tracking store
    first, only to hand the download to the same repository this reaches directly. On a
    `file://` location — runs written before the tracking server, or any SQLite sweep —
    the URI form also skips the copy entirely and returns the path where it already lies.
    A `mlflow-artifacts:` location downloads through the server unless
    `MLFLOW_LOCAL_ARTIFACT_ROOT` says the store is on this filesystem (see `artifact_path`
    and the module docstring), so the tracking URI must be set (see `tracking_uri_from_env`)
    even though no `run_id` is passed.
    """
    frame = pd.read_parquet(artifact_path(artifact_uri, f"metrics/{run_id}.parquet"))
    frame.insert(0, "run_id", run_id)
    return frame


def read_scores_artifact(run_id: str, artifact_uri: str) -> pd.DataFrame:
    """The per-image frame written by `write_predictions`, with `run_id` prepended.

    One row per image: `label`, `score`, `severity` and the rest of the columns a query
    expression can select on. Both readers of this module prepend `run_id`, so a
    concatenation over runs is keyed without the caller tracking which frame came from
    where — `get_scores` and `synevad.analysis.uncertainty` rely on that, and the severity
    recompute simply ignores the extra column.

    Resolved through `artifact_path`, so a store sitting on this filesystem is read in
    place rather than downloaded (see the module docstring). Only `scores.csv` is read,
    never the sibling `anomaly_maps.npz` — the maps are 348 MB per run and nothing that
    resamples image-level scores needs them.
    """
    frame = pd.read_csv(artifact_path(artifact_uri, "predictions/scores.csv"))
    frame.insert(0, "run_id", run_id)
    return frame


def get_ad_metrics(
    runs: pd.DataFrame, *, on_missing: Literal["skip", "raise"] = "skip"
) -> pd.DataFrame:
    """Concatenated per-query metrics for every run in a `get_metadata_by_version` frame.

    Takes the frame rather than a version string because the fast artifact lookup needs
    `artifact_uri`. Versions before the parquet writer landed have no metrics artifact at
    all, so `on_missing="skip"` warns once with the tally instead of raising per run.
    """
    frames, missing = [], []
    for _, run in runs.iterrows():
        try:
            frames.append(read_metrics_artifact(run["run_id"], run["artifact_uri"]))
        except (MlflowException, FileNotFoundError):
            if on_missing == "raise":
                raise
            missing.append(run["run_id"])

    if missing:
        head = ", ".join(missing[:3])
        warnings.warn(
            f"no metrics artifact for {len(missing)}/{len(runs)} runs (e.g. {head})",
            stacklevel=2,
        )

    if not frames:
        return pd.DataFrame(columns=["run_id", *AD_METRIC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def get_scores(
    runs: pd.DataFrame, *, on_missing: Literal["skip", "raise"] = "skip"
) -> pd.DataFrame:
    """Concatenated per-image scores for every run in a `get_metadata_by_version` frame.

    The `get_ad_metrics` twin, one grain finer: that one returns the metrics a run
    computed, this one the predictions they were computed from. Pass the subset of runs
    actually wanted — a full sweep is a few hundred runs of a few hundred rows each, and
    on a tracking-server store every one is a separate artifact download unless
    `MLFLOW_LOCAL_ARTIFACT_ROOT` says otherwise (see the module docstring).

    Runs from before the predictions artifact landed have none at all, so
    `on_missing="skip"` warns once with the tally rather than raising per run.
    """
    frames, missing = [], []
    for _, run in runs.iterrows():
        try:
            frames.append(read_scores_artifact(run["run_id"], run["artifact_uri"]))
        except (MlflowException, FileNotFoundError):
            if on_missing == "raise":
                raise
            missing.append(run["run_id"])

    if missing:
        head = ", ".join(missing[:3])
        warnings.warn(
            f"no predictions artifact for {len(missing)}/{len(runs)} runs (e.g. {head})",
            stacklevel=2,
        )

    if not frames:
        return pd.DataFrame(columns=["run_id", *SCORE_COLUMNS])
    return pd.concat(frames, ignore_index=True)
