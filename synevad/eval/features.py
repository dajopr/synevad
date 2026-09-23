"""Eval features held across the sweep configs that would recompute them identically.

The eval sets are the same images for every model, and PatchCore's embedding of them
depends only on the feature stage — category, backbone, image size, taps, patch size,
embed dimension — never on the coreset ratio, the training subset or the seed. The sweep
varies those three 36 times per feature stage, so 35 of every 36 configs decode,
transform and forward the same images to the same numbers.

The cache holds one key at a time. `sweep.yaml` orders its keys so that the configs
sharing a stage are contiguous, which is what makes a single entry sufficient and keeps
the resident footprint to one group's features. `feature_groups` exists to notice when
that ordering has been undone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from synevad.backbones import resolve_layers
from synevad.data import Batch


@dataclass(frozen=True)
class EvalFeatureKey:
    """Everything a cached eval feature depends on.

    Built from raw config values with only `layers: default` resolved, so it is
    conservative: two configs that would in fact embed identically may still get
    different keys, costing one extraction, but two different feature stages can never
    collide. Reading `layers` through `resolve_layers` rather than `config.model.layers`
    also makes the key independent of `make_patchcore_model` resolving that key in place.
    """

    category: str
    backbone: str
    image_size: tuple[int, ...]
    layers: tuple[int | str, ...]
    patchsize: int | None
    # Both halves of the width: the literal override and the factor scaling it. The
    # per-backbone anchor the factor multiplies is a function of `backbone`, which is
    # already part of the key, so the raw config values are enough to keep two stages apart.
    embed_dimension: int | None
    embed_dimension_factor: float | None
    l2_normalize: bool | None
    half_precision: bool | None
    # Canonical JSON rather than a nested structure so the key stays hashable. Batch size
    # belongs here: cached entries are batch-aligned, so re-batching invalidates them.
    datasets: str

    @classmethod
    def from_config(cls, config: DictConfig) -> "EvalFeatureKey":
        model, data = config.model, config.data
        datasets = {
            "root": data.root,
            "batch_size": data.batch_size,
            "real": bool(data.get("real", True)),
            "synthetic": OmegaConf.to_container(data.synthetic, resolve=True),
        }
        return cls(
            category=str(config.category),
            backbone=str(model.backbone),
            image_size=tuple(data.image_size),
            layers=tuple(resolve_layers(str(model.backbone), model.get("layers"))),
            patchsize=model.get("patchsize"),
            embed_dimension=model.get("embed_dimension"),
            embed_dimension_factor=model.get("embed_dimension_factor"),
            l2_normalize=model.get("l2_normalize"),
            half_precision=model.get("half_precision"),
            datasets=json.dumps(datasets, sort_keys=True, default=str),
        )


@dataclass
class CachedBatch:
    """One eval batch's metadata and patch features, without the decoded images.

    `batch.image` is dropped: it is the largest field, nothing downstream of the backbone
    reads it, and it would otherwise stay resident for the whole group.
    """

    batch: Batch
    features: np.ndarray  # (batch_size * Hp * Wp, D) float32, host memory
    patch_shapes: list[list[int]]
    batch_size: int


@dataclass
class EvalFeatureCache:
    """A single-entry cache of one feature stage's eval sets.

    Deliberately not an LRU: given the key order in `sweep.yaml` one entry is sufficient,
    and a second entry would double a multi-gigabyte footprint to buy nothing.
    """

    key: EvalFeatureKey | None = None
    sets: dict[str, list[CachedBatch]] = field(default_factory=dict)

    def get(self, key: EvalFeatureKey) -> dict[str, list[CachedBatch]] | None:
        return self.sets if self.key == key and self.sets else None

    def put(self, key: EvalFeatureKey, sets: dict[str, list[CachedBatch]]) -> None:
        self.key, self.sets = key, sets

    def clear(self) -> None:
        self.key, self.sets = None, {}


def extract_eval_features(
    model, dataloaders: dict[str, DataLoader]
) -> dict[str, list[CachedBatch]]:
    """Embed every eval set once, keeping the metadata each batch carries."""
    extracted: dict[str, list[CachedBatch]] = {}
    for set_name, dataloader in dataloaders.items():
        entries: list[CachedBatch] = []
        for batch in dataloader:
            batch_size = batch.image.shape[0]
            with torch.no_grad():
                # `detach=False` returns the whole (B*Hp*Wp, D) tensor. The default path
                # iterates it row-wise into one numpy array per patch — 32768 device
                # syncs for a batch of 32 at 256px, which `_predict` then undoes with a
                # single `np.asarray`.
                features, patch_shapes = model._embed(
                    batch.image.to(torch.float).to(model.device),
                    detach=False,
                    provide_patch_shapes=True,
                )
            # The aggregator runs under `no_grad` and the pooling after it has no
            # parameters, so there is no graph to retain. Asserted rather than commented:
            # if upstream ever moves that boundary, the cache would quietly pin one.
            assert not features.requires_grad, "embedded features carry an autograd graph"

            batch.image = None
            entries.append(
                CachedBatch(
                    batch=batch,
                    features=features.cpu().numpy(),
                    patch_shapes=patch_shapes,
                    batch_size=batch_size,
                )
            )
        extracted[set_name] = entries
    return extracted


def score_cached_batch(model, cached: CachedBatch) -> tuple[list, list]:
    """The tail of `PatchCore._predict`, fed features that were embedded earlier.

    Duplicated from `patchcore.patchcore.PatchCore._predict` rather than called: that
    method embeds and scores in one pass with no seam between them, and `patchcore` is
    kept byte-identical to upstream so a rebase stays trivial (see the module docstring of
    `synevad.backbones`). `test_eval_features.py` asserts this stays equal to `_predict`,
    so a patchcore bump fails loudly instead of drifting.
    """
    features = np.asarray(cached.features)
    batch_size = cached.batch_size

    patch_scores = image_scores = model.anomaly_scorer.predict([features])[0]

    image_scores = model.patch_maker.unpatch_scores(image_scores, batchsize=batch_size)
    image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    image_scores = model.patch_maker.score(image_scores)

    patch_scores = model.patch_maker.unpatch_scores(patch_scores, batchsize=batch_size)
    scales = cached.patch_shapes[0]
    patch_scores = patch_scores.reshape(batch_size, scales[0], scales[1])

    masks = model.anomaly_segmentor.convert_to_segmentation(patch_scores)

    return [score for score in image_scores], [mask for mask in masks]


def group_by_feature_key(configs: Iterable[DictConfig]) -> list[list[DictConfig]]:
    """Runs of consecutive configs that share an `EvalFeatureKey`.

    The unit of cache reuse and of multi-GPU sharding: a group must stay intact on one
    worker so the single-entry cache still hits.
    """
    groups: list[list[DictConfig]] = []
    previous: EvalFeatureKey | None = None
    for config in configs:
        key = EvalFeatureKey.from_config(config)
        if key == previous:
            groups[-1].append(config)
        else:
            groups.append([config])
            previous = key
    return groups


def feature_groups(configs: Iterable[DictConfig]) -> list[int]:
    """Sizes of the runs of consecutive configs that share an `EvalFeatureKey`.

    One entry per config means the sweep keys are ordered so that no two neighbours share
    a feature stage, and the cache will miss every time. That is a property of key order
    in `sweep.yaml`, which is exactly the kind of thing a later edit undoes silently.
    """
    return [len(group) for group in group_by_feature_key(configs)]


def shard_feature_groups(
    groups: list[list[DictConfig]], n_workers: int
) -> list[list[DictConfig]]:
    """Round-robin whole feature groups onto workers, then flatten each shard in order.

    Splitting inside a group would force every worker to re-extract the same features.
    Round-robin keeps shards balanced when groups are equal size (the usual case).
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    shards: list[list[DictConfig]] = [[] for _ in range(n_workers)]
    for i, group in enumerate(groups):
        shards[i % n_workers].extend(group)
    return shards
