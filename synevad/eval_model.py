import argparse
import random
import warnings
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from synevad.data import (
    collate_fn,
    get_default_transforms,
    make_mvtecad_dataset,
    make_synevad_dataset,
    train_collate_fn,
)
from synevad.db import (
    check_tracking_store,
    end_run_on_termination,
    is_tracking_server_uri,
    redact_tracking_uri,
    setup_mlflow,
    tracking_uri_from_config,
)
from synevad.eval import (
    EvalFeatureCache,
    EvalFeatureKey,
    evaluate,
    extract_eval_features,
    filter_completed_configs,
    group_by_feature_key,
    shard_feature_groups,
)
from synevad.model import make_patchcore_model

load_dotenv()


def set_seed(seed: int = 42):
    # 1. Python's built-in random module
    random.seed(seed)

    # 2. NumPy's random module
    np.random.seed(seed)

    # 3. PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU / Distributed setups


def setup_and_evaluate(config: DictConfig, cache: EvalFeatureCache | None = None):
    transform = get_default_transforms(size=config.data.image_size)

    # Nothing may go between these two lines. `make_mvtecad_dataset` draws the training
    # subset with `np.random.choice` under the global numpy seed, so work inserted here
    # silently changes which images every model in the sweep trains on — see the
    # assumption `synevad.analysis.selection` documents.
    set_seed(int(config.seed))
    train_dataloader = make_train_dataloader(config, transform)

    model = make_patchcore_model(config, train_dataloader)

    key = EvalFeatureKey.from_config(config)
    features = cache.get(key) if cache is not None else None
    if features is None:
        if cache is not None:
            # Free the previous group before extracting, never after: holding both is
            # what would make the footprint a problem.
            cache.clear()
        features = extract_eval_features(model, make_val_dataloaders(config, transform))
        if cache is not None:
            cache.put(key, features)

    # Run validation and log to MLFlow
    setup_mlflow(config)
    for setname, cached in features.items():
        evaluate(model, cached, setname, config)


def expand_grid(sweep_config: DictConfig) -> list[dict[str, Any]]:
    keys = list(sweep_config.keys())
    grid = [
        {str(key): val for key, val in zip(keys, vals)}
        for vals in product(*(sweep_config[k] for k in keys))
    ]

    return grid


def apply_overrides(base_config: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    config = base_config.copy()
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=False)
        OmegaConf.update(config, f"params.{key.replace('.', '_')}", value)

    return config


def override_experiment(config: DictConfig, name: str | None) -> DictConfig:
    """Point the sweep at a different MLflow experiment; interpolations follow."""
    if not name:
        return config
    cfg = config.copy()
    OmegaConf.update(cfg, "experiment_name", name, merge=False)
    return cfg


def parse_gpus(raw: str | None) -> list[int] | None:
    """Parse `--gpus 0,1,2` into one worker slot per entry. `None` keeps the sequential
    single-process path.

    A device id may repeat: `--gpus 0,0,1` is three workers, two of them sharing card 0.
    Co-locating is worth doing because a worker is nowhere near saturating a card on its
    own — the coreset index and the cached eval features both live in host memory
    (`FaissNN(False, ...)` in `synevad.model`, `.cpu().numpy()` in `extract_eval_features`),
    so what a worker holds on the device is the backbone and one batch. The cost is
    host-side instead: each co-located worker keeps its own copy of one feature group,
    which is the multi-gigabyte footprint `synevad.eval.features` sizes for one process.

    The duplicate this used to reject was a typo guard, and giving it up is the price of
    the feature. `main` prints the placement one line per rank before anything spawns, so
    a `--gpus 0,0` that was meant as `0,1` is still visible.
    """
    if raw is None:
        return None
    gpus = [int(part.strip()) for part in raw.split(",") if part.strip() != ""]
    if not gpus:
        raise ValueError("--gpus must list at least one device id")
    return gpus


def spread_slots(gpu_ids: Sequence[int], n_slots: int) -> list[int]:
    """The first `n_slots` worker slots of `gpu_ids`, dealt round-robin across the cards.

    There is never a point in more workers than feature groups, since a group is never
    split across workers and the surplus ranks would idle. Which slots survive that cap is
    what stops being obvious once ids repeat: a prefix of `--gpus 0,0,1,1` capped to two
    puts both workers on card 0 and leaves card 1 empty. Dealing one slot per distinct
    card per round fills the cards before it doubles up on any of them, and reproduces the
    prefix exactly when every id is distinct.
    """
    remaining = Counter(gpu_ids)
    cards = list(dict.fromkeys(gpu_ids))  # distinct, in the order `--gpus` named them
    slots: list[int] = []
    while len(slots) < n_slots and remaining.total():
        for card in cards:
            if len(slots) == n_slots:
                break
            if remaining[card]:
                slots.append(card)
                remaining[card] -= 1
    return slots


def assign_devices(configs: Sequence[DictConfig], device: str) -> list[DictConfig]:
    assigned: list[DictConfig] = []
    for config in configs:
        cfg = config.copy()
        cfg.device = device
        assigned.append(cfg)
    return assigned


def run_sweep(configs: Sequence[DictConfig]) -> None:
    cache = EvalFeatureCache()
    for i, sweep_config in enumerate(configs):
        print(
            f"[{i + 1}/{len(configs)}] category={sweep_config.category} "
            f"backbone={sweep_config.model.backbone} "
            f"image_size={list(sweep_config.data.image_size)} "
            f"device={sweep_config.device}"
        )
        setup_and_evaluate(sweep_config, cache=cache)


def _worker(rank: int, gpu_ids: list[int], shards: list[list[DictConfig]]) -> None:
    # Before any run opens: when one worker raises, `mp.spawn` SIGTERMs the rest, and
    # only this closes out the run each of them had open.
    end_run_on_termination()
    device = f"cuda:{gpu_ids[rank]}"
    configs = assign_devices(shards[rank], device)
    print(f"worker rank={rank} device={device} configs={len(configs)}")
    run_sweep(configs)


def require_concurrent_tracking(config: DictConfig) -> str:
    """The tracking URI, once it is known to accept concurrent writers.

    Both checks belong in the parent, before the spawn. A store the workers write to
    directly has no one to serialize them (SQLite answers `database is locked`), and an
    unreachable server would surface as N interleaved tracebacks instead of the one line
    that says what is wrong.
    """
    uri = tracking_uri_from_config(config)
    if not is_tracking_server_uri(uri):
        raise SystemExit(
            "multi-GPU sweeps write through an MLflow tracking server; set "
            "MLFLOW_TRACKING_URI to its address (e.g. http://localhost:5000). "
            f"Resolved tracking URI addresses a store directly: {redact_tracking_uri(uri)}"
        )

    try:
        check_tracking_store(uri)
    except Exception as exc:
        raise SystemExit(
            f"cannot reach the MLflow tracking store at {redact_tracking_uri(uri)}: {exc}"
        ) from exc
    return uri


def require_available_gpus(gpu_ids: Sequence[int]) -> None:
    """Reject device ids this host does not have, before a worker is spawned onto one.

    Otherwise the only symptom is a CUDA error raised from inside a worker, after the
    grid is built and its siblings are already training.
    """
    if not torch.cuda.is_available():
        raise SystemExit("--gpus was given but torch.cuda.is_available() is False")
    count = torch.cuda.device_count()
    unknown = sorted({gpu for gpu in gpu_ids if not 0 <= gpu < count})
    if unknown:
        raise SystemExit(
            f"--gpus lists device ids {unknown}, but this host has {count} CUDA "
            f"device(s) (valid ids 0-{count - 1})"
        )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Synevad PatchCore hyperparameter sweep"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "mvtec.yaml"),
        help="Eval YAML (default: synevad/config/mvtec.yaml). "
        "Use synevad/config/ablate_split.yaml or draem_split.yaml for those "
        "synthetic arms.",
    )
    parser.add_argument(
        "--sweep",
        default=str(Path(__file__).parent / "config" / "sweep.yaml"),
        help="Sweep grid YAML (default: synevad/config/sweep.yaml). "
        "Use synevad/config/sweep_draem_split.yaml or sweep_ablate_split.yaml so "
        "resume does not collide with another sweep of the same experiment.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for a parallel sweep (e.g. 0,1,2), one worker per "
        "entry. Repeat an id to co-locate workers on it (e.g. 0,0,1,1 is four workers on "
        "two cards). Omitting this flag keeps the sequential single-process path.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override config experiment_name (and the interpolated logging/tags names). "
        "Use a dedicated name so one synthetic arm is not mixed into another.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip configs the tracking store already has a finished run for on every "
        "eval set. A config whose runs are only partly finished has them deleted and is "
        "re-run in full. --no-resume runs the whole grid.",
    )
    args = parser.parse_args(argv)
    gpu_ids = parse_gpus(args.gpus)
    if gpu_ids is not None:
        require_available_gpus(gpu_ids)

    config = cast(DictConfig, OmegaConf.load(args.config))
    config = override_experiment(config, args.experiment)
    sweep_parameters = cast(DictConfig, OmegaConf.load(args.sweep))

    sweep_grid = [
        apply_overrides(config, overrides)
        for overrides in expand_grid(sweep_parameters)
    ]

    # Before the grouping below, so the cache-miss warning and the shards describe what
    # will actually run, and before anything spawns or touches a GPU.
    if args.resume:
        sweep_grid, n_complete = filter_completed_configs(sweep_grid)
        print(f"[continue] {n_complete} already complete; {len(sweep_grid)} remaining")
        if not sweep_grid:
            print(
                "nothing to run: every config already has a finished run for every "
                "eval set"
            )
            return

    groups = group_by_feature_key(sweep_grid)
    print(f"{len(sweep_grid)} configs in {len(groups)} eval-feature groups")
    if len(groups) == len(sweep_grid) > 1:
        warnings.warn(
            "no two consecutive sweep configs share an eval feature stage, so the "
            "feature cache will miss on every config. Order sweep.yaml so category, "
            "model.backbone and data.image_size vary more slowly than the model knobs "
            "— `expand_grid` varies the last key fastest.",
            stacklevel=2,
        )

    # Capped before the branch below, so a `--gpus` that comes out naming a single worker
    # takes the sequential path: one process has nothing to race with, and sending it
    # through the tracking gate would refuse a sweep that is fine against SQLite.
    if gpu_ids is not None and groups:
        gpu_ids = spread_slots(gpu_ids, len(groups))

    if gpu_ids is None or len(gpu_ids) == 1:
        if gpu_ids is not None:
            sweep_grid = assign_devices(sweep_grid, f"cuda:{gpu_ids[0]}")
        run_sweep(sweep_grid)
        return

    uri = require_concurrent_tracking(config)
    print(f"tracking store: {redact_tracking_uri(uri)}")

    shards = shard_feature_groups(groups, len(gpu_ids))
    for rank, shard in enumerate(shards):
        print(f"shard rank={rank} gpu={gpu_ids[rank]} configs={len(shard)}")

    # Spawn before any CUDA init in the parent so each worker owns its device cleanly.
    mp.spawn(
        _worker,
        args=(gpu_ids, shards),
        nprocs=len(gpu_ids),
        join=True,
    )


def make_train_dataloader(config: DictConfig, transform: T.Compose) -> DataLoader:
    train_dataset = make_mvtecad_dataset(
        config.data.root,
        config.category,
        transform=transform,
        split="train",
        num_train_samples=config.data.num_train_samples,
        # Picks the seed's pool in a pooled split; ignored by a plain MVTec tree, where
        # the seed only reaches the draw through the global RNG set just above.
        seed=int(config.seed),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=config.data.num_workers,
    )
    return train_dataloader


def make_val_dataloaders(
    config: DictConfig, transform: T.Compose
) -> dict[str, DataLoader]:
    # These keys become one MLflow run each, so `synevad.eval.resume.expected_set_names`
    # rebuilds the same set to decide whether a config is already done. An eval set added
    # here without a matching `data.synthetic` entry would be skipped by the resume check.
    dataloaders: dict[str, DataLoader] = {}

    # `data.real: false` is for a corpus with no photographed defects (the Pelton frames):
    # a test set of defect-free images alone has one class, so every real-arm metric is
    # undefined and the run it would log carries nothing.
    if config.data.get("real", True):
        real_val_dataset = make_mvtecad_dataset(
            config.data.root,
            config.category,
            transform=transform,
            split="val",
        )
        dataloaders["real"] = DataLoader(
            real_val_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    for name, dataset_args in config.data.synthetic.items():
        # `category` only seeds the `max_defects` draw, so that one cap drawn per category
        # stays the same set of defects across the sweep. The entry may name it itself for
        # a set whose corpus is not this config's category.
        dataset = make_synevad_dataset(
            **{"category": str(config.category), **dataset_args}, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
        dataloaders[name] = dataloader
    return dataloaders


if __name__ == "__main__":
    main()
