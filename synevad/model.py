from typing import Any, NamedTuple, Protocol

from omegaconf import DictConfig

from torch import nn
from torch.utils.data import DataLoader
from patchcore.common import FaissNN
from patchcore.patchcore import PatchCore
from patchcore.sampler import ApproximateGreedyCoresetSampler

from synevad.backbones import (
    cnn_embed_dimension,
    is_dino_backbone,
    make_dino_backbone,
    resolve_layers,
    resolve_torchvision_backbone,
)

# What the CNN path has always used; DINO defaults to the last transformer block.
CNN_PATCHSIZE = 3


class SynevadModel(Protocol):
    def fit(self, training_data) -> None: ...

    def predict(self, data) -> Any: ...


class FeatureStage(NamedTuple):
    """The `PatchCore.load` arguments that depend on the backbone."""

    backbone: nn.Module
    # As configured (module names or block indices) with `default` resolved, versus the
    # module paths PatchCore hooks. They differ for DINO, whose blocks are tapped by proxy.
    layers: list[int] | list[str]
    layers_to_extract_from: list[str]
    patchsize: int
    embed_dimension: int


def resolve_embed_dimension(config: DictConfig, native: int) -> int:
    """The dimension to project this backbone's features to.

    A width is only meaningful relative to the backbone that produced it: 1024 is
    wide_resnet50_2's layer3 but nearly three times what a dinov3_vits16 token carries, so
    one literal `model.embed_dimension` swept across backbones asks the small ones to
    upsample 384 numbers into 1024 and the CNNs to keep their own. `native` is therefore
    the per-backbone anchor (a CNN's deepest default tap, a ViT's `embed_dim`) and
    `model.embed_dimension_factor` scales it, so a single sweep value means "three quarters
    of whatever this backbone has" — 288 for dinov3_vits16, 768 for wide_resnet50_2.

    `model.embed_dimension` still overrides the anchor with a literal width, for the case
    where a run must pin one number across backbones; the factor then scales that instead.
    """
    base = config.model.get("embed_dimension", None) or native
    factor = float(config.model.get("embed_dimension_factor", 1.0))
    if factor <= 0:
        raise ValueError(f"`embed_dimension_factor` must be positive, got {factor}")
    # Rounded, not truncated, so a factor of 1/3 on 384 gives 128 rather than 127; a factor
    # small enough to round to 0 would make `adaptive_avg_pool1d` raise deep in PatchCore.
    return max(1, round(base * factor))


def _feature_stage(config: DictConfig) -> FeatureStage:
    name = config.model.backbone
    layers = resolve_layers(name, config.model.get("layers"))

    if not is_dino_backbone(name):
        return FeatureStage(
            backbone=resolve_torchvision_backbone(name),
            layers=layers,
            layers_to_extract_from=layers,  # type: ignore[arg-type]
            patchsize=config.model.get("patchsize", CNN_PATCHSIZE),
            embed_dimension=resolve_embed_dimension(config, cnn_embed_dimension(name)),
        )

    backbone = make_dino_backbone(
        name,
        layers=layers,  # type: ignore[arg-type]
        image_size=config.data.image_size,
        l2_normalize=config.model.get("l2_normalize", True),
        half_precision=config.model.get("half_precision", False),
    )
    return FeatureStage(
        backbone=backbone,
        layers=layers,
        layers_to_extract_from=backbone.layer_names,
        # patchsize 1 = no neighbourhood pool; raw token dim (factor 1) = no projection.
        patchsize=config.model.get("patchsize", 1),
        embed_dimension=resolve_embed_dimension(config, backbone.embed_dim),
    )


def make_patchcore_model(config: DictConfig, pc_dataloader: DataLoader) -> SynevadModel:

    model = PatchCore(device=config.device)
    sampler = ApproximateGreedyCoresetSampler(
        config.model.coreset_sampling_ratio,
        config.device,
        config.model.number_of_starting_points,
        config.model.dimension_to_project_features_to,
    )
    stage = _feature_stage(config)
    # `params.layers` interpolates `model.layers`, and logging happens after this, so a
    # sweep over backbones records the taps it used instead of "default" on every run.
    config.model.layers = stage.layers

    model.load(
        backbone=stage.backbone,
        layers_to_extract_from=stage.layers_to_extract_from,
        device=config.device,
        input_shape=(3, config.data.image_size[0], config.data.image_size[1]),
        patchsize=stage.patchsize,
        pretrain_embed_dimension=stage.embed_dimension,
        target_embed_dimension=stage.embed_dimension,
        featuresampler=sampler,  # type: ignore
        anomaly_score_num_nn=config.model.anomaly_score_num_nn,
        # `PatchCore.load`'s default is `FaissNN(False, 4)` evaluated at import time — a
        # mutable default argument, so every model in a sweep would share one index. It
        # works today only because each `fit` resets it; passing one per model removes
        # the coupling before something keeps two models alive at once.
        nn_method=FaissNN(False, config.model.get("faiss_num_workers", 4)),
    )

    model.fit(pc_dataloader)
    return model
