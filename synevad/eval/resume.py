"""Skipping the sweep configs the tracking store already holds a full result for.

A sweep costs GPU-days and gets interrupted. Re-running a config that already logged a
finished run for every eval set is pure waste; but *half* skipping one is worse than not
skipping it at all, because the arm that never ran is then never run, and `pair_runs`
drops the whole sweep point when it cannot pair a real run with its synthetic sibling.
So "complete" is all-or-nothing per config, and every uncertainty here resolves toward
re-running.

Identity is `declared_tags` plus the full `declared_params` vector, compared as whole
sorted tuples of stored strings. The tags look redundant with the params — `apply_overrides`
aliases every swept key into `params.*` — but only for keys that happen to be swept. Drop
`category` from `sweep.yaml` and the params vector stops telling categories apart, and 14
of 15 would be wrongly skipped. `synevad.analysis.default_group_keys` is the sibling with
the weaker contract: it may fall back to params alone and says nothing about extra params
on the stored side, which is fine for grouping runs that already exist and wrong for
deciding not to produce one.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from omegaconf import DictConfig

from synevad.backbones import resolve_layers
from synevad.db import (
    declared_params,
    declared_tags,
    get_experiment_runs,
    is_tracking_server_uri,
    redact_tracking_uri,
    tracking_uri_from_config,
)

# The eval set `make_val_dataloaders` always builds, alongside one per `data.synthetic`
# entry. Kept in sync with `synevad.analysis.data.REAL_SET_NAME`, which names the same set
# on the read side.
REAL_SET_NAME = "real"

# Tags `get_tags` adds per eval set, and `summarize_metrics` per outcome. They vary across
# the runs of ONE config, so including them would make every config look incomplete.
# Subtracted from the declared names rather than assumed absent: a config that declared
# `tags.set_name` itself would otherwise match nothing, forever.
RUNTIME_TAG_NAMES: frozenset[str] = frozenset(
    {"set_name", "severity", "scorer_result", "validity"}
)

FINISHED = "FINISHED"

# A run another process is still writing. Never counted as complete (it has no metrics
# yet) and never deleted (it is not ours to delete).
RUNNING = "RUNNING"


@dataclass(frozen=True)
class RunIdentity:
    """What distinguishes one sweep point from another, as the strings MLflow stores.

    Both members are the full sorted vector on both sides, which is what makes the
    comparison symmetric: a stored run carrying a param this config does not set is a
    different sweep point, and re-running is the correct answer.
    """

    tags: tuple[tuple[str, str], ...]
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    set_name: str


def expected_set_names(config: DictConfig) -> set[str]:
    """The eval sets `setup_and_evaluate` will log a run for.

    Mirrors the dict `make_val_dataloaders` builds — the real test set (unless
    `data.real: false`) plus one entry per key of `data.synthetic`. An eval set added there
    without a matching config entry would make this judge completeness on the wrong arms.
    """
    real = {REAL_SET_NAME} if config.data.get("real", True) else set()
    return {*real, *config.data.synthetic}


def declared_tag_names(configs: Sequence[DictConfig]) -> tuple[str, ...]:
    """The tag names that take part in the identity: declared by the configs, minus the
    ones a run acquires at eval time."""
    names: set[str] = set()
    for config in configs:
        names.update(config.tags.keys())
    return tuple(sorted(names - RUNTIME_TAG_NAMES))


def as_logged(config: DictConfig) -> DictConfig:
    """`config` with the keys the eval path rewrites in place before it logs.

    `make_patchcore_model` writes the taps it actually hooked back into
    `config.model.layers`, so that a sweep over backbones records `['layer2', 'layer3']`
    rather than `default` on every run — and `params.layers` interpolates that key. This
    check runs before any model is built, so without the same resolution it asks the store
    whether `default` equals `['layer2', 'layer3']`, which is never true: the whole
    `v0_fixed` sweep (4,583 finished runs, 2,290 complete sweep points) was invisible to
    the resume check and every config in the grid looked pending.
    `EvalFeatureKey.from_config` reads `layers` through `resolve_layers` for this reason
    too, and the same argument applies here.

    `layers` is currently the only such key — it is the one thing that differs between the
    config the sweep holds and the params its run carries. A config that declares no model
    at all is returned untouched, so the identity helpers stay usable on a bare config.
    """
    model = config.get("model")
    if model is None or "layers" not in model or "backbone" not in model:
        return config
    resolved = config.copy()
    resolved.model.layers = resolve_layers(str(model.backbone), model.get("layers"))
    return resolved


def config_identity(config: DictConfig, tag_names: Sequence[str]) -> RunIdentity:
    """`config`'s identity, built through the same helpers that write the run."""
    config = as_logged(config)
    tags = declared_tags(config)
    return RunIdentity(
        tags=tuple((name, tags[name]) for name in tag_names if name in tags),
        params=tuple(sorted(declared_params(config).items())),
    )


def index_runs(
    runs: pd.DataFrame, tag_names: Sequence[str]
) -> dict[RunIdentity, list[RunRecord]]:
    """Every run in `runs`, grouped by the sweep point it belongs to."""
    index: dict[RunIdentity, list[RunRecord]] = {}
    if "tags.set_name" not in runs.columns or "status" not in runs.columns:
        # The shape `search_runs` returns for an experiment that does not exist yet: the
        # six info columns and nothing else. Reaching for `params.*` here would raise
        # instead of skipping nothing.
        return index

    param_columns = [col for col in runs.columns if col.startswith("params.")]
    tag_columns = [
        (name, f"tags.{name}") for name in tag_names if f"tags.{name}" in runs.columns
    ]

    for row in runs.to_dict("records"):
        # A column exists because *some* run in the frame has it; the ones that do not get
        # a null. Dropping nulls rather than keeping them as values is what makes each
        # run's vector "the params this run actually has", and the comparison symmetric.
        identity = RunIdentity(
            tags=tuple(
                (name, str(row[col]))
                for name, col in tag_columns
                if not pd.isna(row[col])
            ),
            params=tuple(
                sorted(
                    (col.removeprefix("params."), str(row[col]))
                    for col in param_columns
                    if not pd.isna(row[col])
                )
            ),
        )
        index.setdefault(identity, []).append(
            RunRecord(
                run_id=str(row["run_id"]),
                status=str(row["status"]),
                set_name=str(row["tags.set_name"]),
            )
        )
    return index


def partition_grid(
    configs: Sequence[DictConfig],
    index: dict[RunIdentity, list[RunRecord]],
    tag_names: Sequence[str],
) -> tuple[list[DictConfig], list[str]]:
    """`configs` that still need running, and the run ids left over from earlier attempts.

    A config is complete when every expected set name has a FINISHED run. Its stale runs
    are then just the failed leftovers. A config that is *not* complete has all of its
    runs deleted, the finished ones included: re-running it writes a second copy of every
    arm otherwise, and the analysis then has to guess which pair belongs together.
    """
    pending: list[DictConfig] = []
    stale: list[str] = []
    n_running = 0

    for config in configs:
        records = index.get(config_identity(config, tag_names), [])
        finished = {rec.set_name for rec in records if rec.status == FINISHED}
        n_running += sum(rec.status == RUNNING for rec in records)

        if expected_set_names(config) <= finished:
            stale.extend(
                rec.run_id
                for rec in records
                if rec.status not in (FINISHED, RUNNING)
            )
            continue

        pending.append(config)
        stale.extend(rec.run_id for rec in records if rec.status != RUNNING)

    if n_running:
        warnings.warn(
            f"{n_running} run(s) matching this grid are still RUNNING; they were left "
            "alone and do not count as complete. Another sweep may be writing to this "
            "store.",
            stacklevel=2,
        )
    return pending, stale


def delete_runs(run_ids: Sequence[str]) -> None:
    """Soft-delete `run_ids`. `mlflow runs restore <id>` brings one back."""
    for run_id in run_ids:
        mlflow.delete_run(run_id)


def _sqlite_store_is_absent(uri: str) -> bool:
    """Whether `uri` names a SQLite file that does not exist yet.

    Reading through MLflow would instantiate the store, which creates the file and runs
    its migrations — and on a first-ever sweep `MLFLOW_DIR` may not exist at all, since
    `artifact_location` is what makes it, later. A store that is not there holds no runs,
    so there is nothing to ask it.
    """
    if is_tracking_server_uri(uri):
        return False
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return False
    return not Path(uri[len(prefix) :]).exists()


def fetch_runs(configs: Sequence[DictConfig]) -> tuple[pd.DataFrame, str]:
    """Every run that could belong to this grid, plus the URI it came from.

    Unfinished runs are included: the caller deletes the leftovers of interrupted configs,
    which means it has to see the KILLED and FAILED rows too.
    """
    uri = tracking_uri_from_config(configs[0])
    # Process-wide on purpose, and not cleaned up afterwards: `setup_mlflow` sets the same
    # URI before the first run opens, and `mp.spawn` children set it themselves.
    mlflow.set_tracking_uri(uri)

    if _sqlite_store_is_absent(uri):
        return pd.DataFrame(), uri

    experiment_names = sorted({str(c.logging.experiment_name) for c in configs})
    # MLflow filter strings have no OR, and tag comparators have no IN, so narrowing by
    # version server-side costs one call per distinct version. `sweep.yaml` sweeps a
    # single version, so that is one call for the whole grid.
    versions = sorted({str(c.tags.version) for c in configs if "version" in c.tags})
    filters = [f"tags.version='{version}'" for version in versions] or [""]

    try:
        frames = [
            get_experiment_runs(
                experiment_names, filter_string=filter_string, finished_only=False
            )
            for filter_string in filters
        ]
    except Exception as exc:
        # Failing closed. Warning and running the grid anyway turns a typo'd
        # MLFLOW_TRACKING_URI into a GPU-week.
        raise SystemExit(
            f"cannot read the tracking store at {redact_tracking_uri(uri)} to find "
            f"which configs are already done: {exc}. Pass --no-resume to run the whole "
            "grid without this check."
        ) from exc

    runs = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return runs, uri


def filter_completed_configs(
    configs: Sequence[DictConfig], *, delete_stale: bool = True
) -> tuple[list[DictConfig], int]:
    """`configs` minus the ones already complete, in order, with the skipped count.

    Order is preserved because `group_by_feature_key` only ever compares a config with its
    immediate predecessor: dropping elements leaves the survivors adjacent, so groups can
    shrink or merge but never split.
    """
    if not configs:
        return [], 0

    tag_names = declared_tag_names(configs)
    runs, uri = fetch_runs(configs)
    index = index_runs(runs, tag_names)
    pending, stale = partition_grid(configs, index, tag_names)

    print(f"[continue] {len(runs)} run(s) in {redact_tracking_uri(uri)}")
    if stale and delete_stale:
        delete_runs(stale)
        print(
            f"[continue] deleted {len(stale)} run(s) left over from interrupted configs "
            "(soft delete: `mlflow runs restore <id>` brings one back)"
        )
    return pending, len(configs) - len(pending)
