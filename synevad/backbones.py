"""DINOv2/DINOv3 patch tokens as a PatchCore backbone, with upstream PatchCore unmodified.

`patchcore.common.NetworkFeatureAggregator` hooks *named submodules* and expects every hook
to fire with `B x C x H x W`. A ViT has neither the layer names nor the spatial layout, so
`DinoBackbone` owns a `ModuleDict` of `nn.Identity` *taps* — one per requested block — and
pushes each reshaped token grid through its tap. PatchCore's hooks then fire as usual, on a
module path (`taps.l-1`) its name resolver already handles.

`patchsize=1` skips the 3x3 neighbourhood pool, matching `pretrain_embed_dimension` /
`target_embed_dimension` to `embed_dim` turns `Preprocessing`/`Aggregator` into identity
pools, and `l2_normalize` makes `IndexFlatL2` return `2 - 2cos` — a monotone function of
cosine distance, so rank metrics and the smoothed map are unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import timm
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

# timm reimplements both families as a plain `VisionTransformer`, so one code path covers
# DINOv2 (patch 14) and DINOv3 (patch 16). It also mirrors the DINOv3 weights under its own
# HF org, avoiding the license gate on the `facebook/dinov3-*` repos (spec S2).
DINO_MODELS: dict[str, str] = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2.lvd142m",
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vitsplus16": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
}


def is_dino_backbone(name: str) -> bool:
    return name in DINO_MODELS


# What `layers: default` resolves to. The two families do not share a namespace — block
# indices for a ViT, module names for a CNN — so a sweep over `model.backbone` cannot
# carry one literal `layers` list across both. DINO defaults to the last block; layer2+layer3
# is what the CNN path has always used.
DEFAULT_LAYERS = "default"
DEFAULT_DINO_LAYERS: tuple[int, ...] = (-1,)
DEFAULT_CNN_LAYERS: tuple[str, ...] = ("layer2", "layer3")

# The CNN taps, shallow to deep; both `resolve_torchvision_backbone` models expose these.
CNN_STAGES: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")

# The width PatchCore projects a CNN's concatenated taps to when nothing overrides it: the
# channel count of the deepest tap in the family default. 1024 is layer3 of
# wide_resnet50_2 and PatchCore's own published setting; resnet18 is a quarter as wide
# everywhere (layer2 128, layer3 256), so the same rule gives it 256 rather than a 4x
# upsample of features that never carried that many channels. A ViT has no entry here —
# its native width is the loaded model's `embed_dim`, read off the built backbone.
CNN_EMBED_DIMENSIONS: dict[str, int] = {
    "wide_resnet50_2": 1024,
    "resnet18": 256,
}


def cnn_embed_dimension(name: str) -> int:
    """The native embed dimension of CNN backbone `name`."""
    if name not in CNN_EMBED_DIMENSIONS:
        raise ValueError(f"Backbone {name} not supported!")
    return CNN_EMBED_DIMENSIONS[name]


# What `<n>_additional` widens the family default *with*, nearest first. The two families
# widen in opposite directions, which is the point: the CNN default stops short of the
# network end, so it reaches deeper before it reaches shallower, while a ViT default is
# already the last block and can only reach earlier. Anchoring on the default rather than
# on the end of the network is what keeps every rung a sensible config — naming CNN taps
# from the end instead would propose layer3+layer4, dropping the layer2 that PatchCore
# depends on for spatial detail.
CNN_EXTENSIONS: tuple[str, ...] = ("layer4", "layer1")
_ORDINALS: dict[str, int] = {"one": 1, "two": 2}

# The closed, family-neutral vocabulary. Every member resolves for every backbone, which is
# what lets `model.backbone` and `model.layers` be swept as an unconstrained cartesian
# product. The ladder stops where the shorter family runs out: the CNN has exactly
# `len(CNN_EXTENSIONS)` layers to add, so a third rung would resolve for a ViT and raise for
# a CNN, and the product would no longer be safe.
LAYER_ALIASES: tuple[str, ...] = (DEFAULT_LAYERS, "one_additional", "two_additional")
_ADDITIONAL = re.compile(r"(one|two)_additional")


def resolve_layers(
    name: str, layers: str | Sequence[int] | Sequence[str] | None
) -> list[int] | list[str]:
    """The layers to tap for `name`, resolving the family-neutral aliases.

    `"default"` (or `None`, i.e. a missing config key) gives the family's status-quo taps;
    `"<n>_additional"` widens that default by n layers. Both are backbone-independent, which
    is what lets a sweep vary `model.backbone` and `model.layers` as an unconstrained
    product. A literal list is an explicit, family-specific override.
    """
    if layers is None or layers == DEFAULT_LAYERS:
        defaults = DEFAULT_DINO_LAYERS if is_dino_backbone(name) else DEFAULT_CNN_LAYERS
        return list(defaults)  # type: ignore[arg-type]
    if isinstance(layers, str):
        return _resolve_alias(name, layers)
    return _checked_literal(name, list(layers))


def _resolve_alias(name: str, alias: str) -> list[int] | list[str]:
    """Widen `name`'s family default by N layers, in its own tap namespace."""
    match = _ADDITIONAL.fullmatch(alias)
    if match is None:
        raise ValueError(
            f"`layers` must be a list or one of {list(LAYER_ALIASES)}, got {alias!r}"
        )
    n = _ORDINALS[match.group(1)]
    if is_dino_backbone(name):
        # Negative indices keep this depth-agnostic: `make_dino_backbone` range-checks them
        # against the loaded block count, so one alias fits a 12- and a 24-block ViT alike.
        return list(range(-(n + len(DEFAULT_DINO_LAYERS)), 0))
    widened = {*DEFAULT_CNN_LAYERS, *CNN_EXTENSIONS[:n]}
    return [stage for stage in CNN_STAGES if stage in widened]


def _checked_literal(name: str, layers: list) -> list[int] | list[str]:
    """Reject a literal written in the *other* family's namespace.

    Only the CNN side is checked here: a bad module name would otherwise surface as a
    `KeyError` from inside upstream `patchcore.common.NetworkFeatureAggregator`, hours into
    a sweep. `make_dino_backbone` already rejects empty, repeated and out-of-range indices.
    """
    if is_dino_backbone(name):
        return layers
    unknown = [layer for layer in layers if layer not in CNN_STAGES]
    if unknown:
        if any(isinstance(layer, int) for layer in unknown):
            raise ValueError(
                f"`layers` {unknown} are block indices, which only a DINO backbone takes; "
                f"{name} wants module names from {CNN_STAGES}"
            )
        raise ValueError(
            f"unknown {name} layer(s) {unknown}; expected some of {CNN_STAGES}"
        )
    return layers


def _tap_name(layer: int) -> str:
    # `-` is legal in a `ModuleDict` key (only `.` is reserved) and is not numeric, so
    # negative indices survive the round trip through `NetworkFeatureAggregator`.
    return f"l{layer}"


class DinoBackbone(nn.Module):
    """A timm DINO ViT exposed to PatchCore as a hookable, spatially-shaped backbone."""

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[int],
        patch_size: int,
        embed_dim: int,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.layers = list(layers)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.l2_normalize = l2_normalize
        self.taps = nn.ModuleDict(
            {_tap_name(layer): nn.Identity() for layer in self.layers}
        )

    @property
    def layer_names(self) -> list[str]:
        """Module paths to hand to PatchCore as `layers_to_extract_from`."""
        return [f"taps.{_tap_name(layer)}" for layer in self.layers]

    def forward(self, images: Tensor) -> list[Tensor]:
        # PatchCore casts to float32 in `_embed`, so match the encoder rather than making
        # callers know about `half_precision`.
        dtype = next(self.model.parameters()).dtype

        assert not isinstance(self.model.forward_intermediates, Tensor)

        features = self.model.forward_intermediates(
            images.to(dtype),
            indices=self.layers,
            norm=True,
            return_prefix_tokens=False,  # Drops CLS and register tokens
            output_fmt="NCHW",  # Reshapes 1D patch sequence to BxCxHxW
            intermediates_only=True,
        )
        outputs = []
        for tap, feature in zip(self.taps.values(), features):
            feature = feature.float()  # FAISS takes float32 only
            if self.l2_normalize:
                feature = F.normalize(feature, dim=1)
            # Called for the hook, not the return value: the last tap's hook raises to stop
            # the forward early, so anything after this loop may not run.
            outputs.append(tap(feature))
        return outputs


def make_dino_backbone(
    name: str,
    layers: Sequence[int],
    image_size: int | Sequence[int],
    l2_normalize: bool = True,
    half_precision: bool = False,
    pretrained: bool = True,
) -> DinoBackbone:
    """Build a DINO backbone sized for `image_size` inputs.

    `layers` are block indices, negative counting from the last. `image_size` is a side
    length or (height, width); each side must be a whole number of patches, and the spec's
    default of 448 gives DINOv2 a 32x32 = 1024 token grid. Only tests pass `pretrained`.
    """
    if name not in DINO_MODELS:
        raise ValueError(
            f"Unknown DINO backbone {name!r}; expected one of {sorted(DINO_MODELS)}"
        )
    if not layers:
        raise ValueError("`layers` must name at least one transformer block")
    if len(set(layers)) != len(layers):
        # One tap per index, so a repeat would leave `layer_names` longer than the hooks.
        raise ValueError(f"`layers` must not repeat an index, got {list(layers)}")

    size = (
        (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    )

    model = timm.create_model(
        DINO_MODELS[name],
        pretrained=pretrained,
        num_classes=0,
        img_size=size,
        dynamic_img_size=True,
    )
    patch_size = model.patch_embed.patch_size[0]  # type: ignore
    for side in size:
        if side % patch_size != 0:
            # `dynamic_img_size` would otherwise floor-divide in silence: 256px through a
            # patch-14 model covers only 252 of them.
            raise ValueError(
                f"image_size {tuple(size)} is not a multiple of {name}'s patch size "
                f"{patch_size}; use e.g. {patch_size * (side // patch_size)} or "
                f"{patch_size * (side // patch_size + 1)}"
            )

    depth = len(model.blocks)  # type: ignore
    for layer in layers:
        if not -depth <= layer < depth:
            raise ValueError(
                f"Layer index {layer} out of range for {name}, which has {depth} blocks"
            )

    if half_precision:
        model = model.half()

    return DinoBackbone(
        model=model.eval(),
        layers=layers,
        patch_size=patch_size,
        embed_dim=model.embed_dim,
        l2_normalize=l2_normalize,
    )


def resolve_torchvision_backbone(name: str) -> nn.Module:
    """The pre-existing CNN backbones, unchanged."""
    if name == "wide_resnet50_2":
        return models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    raise ValueError(f"Backbone {name} not supported!")
