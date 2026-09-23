"""Dense visual-change map from frozen backbone features — the mask's seed.

A photometric residual answers "did these pixels change". It cannot answer "did the *content*
change", and an instruct-edit model changes pixels everywhere: it re-renders texture, redraws
grain, shifts a weave by a pixel. Those changes clear a noise-calibrated threshold as readily
as a defect does, and the residual has no way to tell them apart.

Feature-map cosine distance does tell them apart. A uniform re-render moves every position in
much the same direction — which :func:`local_feature_distance` subtracts out — while a defect
changes what one region *is*, and that shows up as a large local distance. Measured on real
crops with a synthetic defect composited over a global re-render (average precision against
known ground truth):

===================  =============  ==============
defect scale         photometric    DINOv3-S/16
===================  =============  ==============
4 px scratch                 0.018           0.119
12 px streak                 0.505           0.546
45 px pit                    0.800           0.930
mixed                        0.633           0.895
===================  =============  ==============

``masks.encoder`` picks the backbone: ``dinov3`` (patch-token cosine, 16 px grid) or
``resnet18`` (ImageNet CNN feature maps; ``layer2`` is 1/8 of the crop). Both share
:func:`local_feature_distance` and the same paste → post-blend → guided-refine wrapper.
Guided refine snaps the seed onto the photometric residual so the stored mask is not stuck
on the feature grid.

DINOv3 scores well on the table above. Two caveats sit next to that number: the token grid
is 16 px (ResNet ``layer2`` is 8 px), and ``synevad`` evaluates DINOv3 PatchCore backbones
(``synevad/config/sweep.yaml``) — ground truth whose geometry came from DINOv3 features
encodes the detector's own representation. ResNet-18 is the independence-preserving
alternative; switch with ``masks.encoder: resnet18``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from PIL import Image

_RESAMPLE = Image.Resampling.LANCZOS


def local_feature_distance(
    clean: np.ndarray,
    edit: np.ndarray,
    *,
    shift_remove: float = 1.0,
    cosine_weight: float = 1.0,
    l2_weight: float = 0.0,
) -> np.ndarray:
    """Per-token local change between L2-normalised grids ``(H, W, C)``.

    ``shift_remove`` subtracts the spatial-median feature delta before scoring, so a
    uniform re-render (every token moved the same way) falls out and only spatially
    *local* differences remain. ``cosine_weight`` / ``l2_weight`` mix ``1 - cos`` with
    Euclidean distance on the (possibly shift-removed) tokens; a zero pair yields cosine.
    """
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(edit, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"token grids must match (H, W, C), got {a.shape} vs {b.shape}")
    shift = float(np.clip(shift_remove, 0.0, 1.0))
    if shift > 0:
        delta = b - a
        global_delta = np.median(delta.reshape(-1, delta.shape[-1]), axis=0)
        b = b - shift * global_delta
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    cosine = 1.0 - np.clip((a * b).sum(axis=-1), -1.0, 1.0)
    cw, lw = float(cosine_weight), float(l2_weight)
    if lw <= 0 and cw <= 0:
        return cosine
    if lw <= 0:
        return cosine
    l2 = np.linalg.norm(a - b, axis=-1)
    if cw <= 0:
        return l2
    return (cw * cosine + lw * l2) / (cw + lw)


def _local_feature_distance_torch(
    clean,
    edit,
    *,
    shift_remove: float = 1.0,
    cosine_weight: float = 1.0,
    l2_weight: float = 0.0,
):
    """Torch twin of :func:`local_feature_distance` for ``(H, W, C)`` tensors."""
    a, b = clean, edit
    shift = float(min(max(shift_remove, 0.0), 1.0))
    if shift > 0:
        delta = b - a
        global_delta = delta.reshape(-1, delta.shape[-1]).median(dim=0).values
        b = b - shift * global_delta
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    cosine = 1.0 - (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    cw, lw = float(cosine_weight), float(l2_weight)
    if lw <= 0:
        return cosine
    l2 = (a - b).norm(dim=-1)
    if cw <= 0:
        return l2
    return (cw * cosine + lw * l2) / (cw + lw)


def combine_layer_maps(
    maps: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Weighted mean of same-shaped 2-D change maps. Empty ``weights`` → uniform."""
    if not maps:
        raise ValueError("combine_layer_maps requires at least one map")
    stacked = np.stack([np.asarray(m, dtype=np.float64) for m in maps], axis=0)
    if weights is None or len(weights) == 0:
        w = np.ones(stacked.shape[0], dtype=np.float64)
    else:
        if len(weights) != stacked.shape[0]:
            raise ValueError(
                f"{len(weights)} weights for {stacked.shape[0]} layer maps"
            )
        w = np.asarray(weights, dtype=np.float64)
        if not np.any(w > 0):
            w = np.ones_like(w)
    w = w / w.sum()
    return np.tensordot(w, stacked, axes=(0, 0))


def upsample_change_map(mag: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample of a token-grid map to ``(width, height)`` PIL size."""
    return np.asarray(
        Image.fromarray(np.asarray(mag, dtype=np.float32), mode="F").resize(
            size, Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    )


# --- encoder selection ------------------------------------------------------

ENCODER_DINO = "dinov3"
ENCODER_RESNET = "resnet18"
_DINO_ALIASES = frozenset({"dinov3", "dino", "dinov3_vits16", "semantic"})
_RESNET_ALIASES = frozenset({"resnet18", "resnet", "resnet8", "cnn"})


def resolve_encoder_kind(masks: dict | None) -> str:
    """``dinov3`` or ``resnet18`` from ``masks.encoder`` (or the older ``masks.arm``)."""
    masks = masks or {}
    raw = str(masks.get("encoder") or masks.get("arm") or ENCODER_RESNET).strip().lower()
    if raw in _DINO_ALIASES:
        return ENCODER_DINO
    if raw in _RESNET_ALIASES:
        return ENCODER_RESNET
    raise ValueError(f"unknown masks.encoder {raw!r}; expected 'dinov3' or 'resnet18'")


def seed_block(masks: dict | None) -> dict:
    """Paste-seed sub-block (``masks.dinov3`` or ``masks.resnet``) for the active encoder."""
    masks = masks or {}
    key = ENCODER_DINO if resolve_encoder_kind(masks) == ENCODER_DINO else "resnet"
    return dict(masks.get(key) or {})


def default_min_change(kind: str, *, post_blend: bool = False) -> float:
    """Absolute floor on the change map. Post-blend is always 0.02; paste is encoder-specific."""
    if post_blend or kind == ENCODER_DINO:
        return 0.02
    return 0.25


# --- DINOv3 change map ------------------------------------------------------

# Patch-token cosine distance. Default is ViT-S/16 (same family as the PatchCore sweep).
# Layers are transformer-block indices, negative from the end; they share a 1/16 grid.
DEFAULT_DINO_MODEL = "dinov3_vits16"
DEFAULT_DINO_LAYERS: tuple[int, ...] = (-1,)
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def resolve_dino_layers(layers: Sequence[int | str] | None) -> tuple[int, ...]:
    """Validate and freeze a block-index tuple; empty / None → :data:`DEFAULT_DINO_LAYERS`."""
    if not layers:
        return DEFAULT_DINO_LAYERS
    try:
        out = tuple(int(n) for n in layers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DINOv3 layers must be transformer-block indices (e.g. [-1]), got {list(layers)}"
        ) from exc
    return out if out else DEFAULT_DINO_LAYERS


def _unique_ints(values: Sequence[int]) -> tuple[int, ...]:
    seen: list[int] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return tuple(seen)


class DinoEncoder:
    """DINOv3 (or DINOv2) patch tokens → cosine-distance change map for a crop pair.

    One forward extracts the configured transformer blocks via ``forward_intermediates``.
    Callers pick a subset and mix with :func:`combine_layer_maps` after each block is
    scored and upsampled to crop size. Clean-crop maps are cached by ``cache_key``.
    """

    def __init__(
        self,
        model: str = DEFAULT_DINO_MODEL,
        *,
        device: str = "cuda:0",
        cache_size: int = 8,
        pretrained: bool = True,
        layers: Sequence[int | str] | None = None,
    ) -> None:
        import timm
        import torch

        from synevad.backbones import DINO_MODELS

        if model not in DINO_MODELS:
            raise ValueError(
                f"unknown DINO backbone {model!r}; expected one of {sorted(DINO_MODELS)}"
            )
        self.name = str(model)
        self.device = device
        self._torch = torch
        self._cache: OrderedDict[str, dict[int, np.ndarray]] = OrderedDict()
        self._cache_size = int(cache_size)
        self.extract_layers = resolve_dino_layers(layers)
        # Dummy construction size; ``dynamic_img_size`` accepts any multiple of the patch.
        print(f"[dinov3] loading {model} on {device} …", flush=True)
        vit = timm.create_model(
            DINO_MODELS[model],
            pretrained=pretrained,
            num_classes=0,
            img_size=256,
            dynamic_img_size=True,
        )
        self.patch_size = int(vit.patch_embed.patch_size[0])  # type: ignore[union-attr]
        depth = len(vit.blocks)  # type: ignore[arg-type]
        for layer in self.extract_layers:
            if not -depth <= layer < depth:
                raise ValueError(
                    f"layer index {layer} out of range for {model}, which has {depth} blocks"
                )
        vit.eval()
        for p in vit.parameters():
            p.requires_grad_(False)
        self._model = vit.to(device).eval()
        landed = next(self._model.parameters()).device
        print(f"[dinov3] {model} ready on {landed}", flush=True)

    def _prepare(self, img: Image.Image):
        """ImageNet-normalised ``(1, 3, H, W)`` tensor, sides rounded up to a patch multiple."""
        torch = self._torch
        rgb = img.convert("RGB")
        w, h = rgb.size
        ps = self.patch_size
        tw = max(ps, ((w + ps - 1) // ps) * ps)
        th = max(ps, ((h + ps - 1) // ps) * ps)
        if (w, h) != (tw, th):
            rgb = rgb.resize((tw, th), _RESAMPLE)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return (
            torch.from_numpy((arr - _IMAGENET_MEAN) / _IMAGENET_STD)
            .permute(2, 0, 1)[None]
            .to(self.device)
        )

    def _all_layer_maps(
        self, img: Image.Image, cache_key: str | None = None
    ) -> dict[int, np.ndarray]:
        """L2-normalised ``(H, W, C)`` maps for every extracted block, optionally memoised."""
        if cache_key is not None and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        feats = self._forward_batch(self._prepare(img))
        stored: dict[int, np.ndarray] = {}
        for idx, feat in feats.items():
            tok = feat[0].permute(1, 2, 0).float().cpu().numpy()
            stored[idx] = tok
        if cache_key is not None:
            self._cache[cache_key] = stored
            while len(self._cache) > max(self._cache_size, 1):
                self._cache.popitem(last=False)
        return stored

    def _forward_batch(self, tensor):
        """L2-normalised ``(B, C, H, W)`` tensors on the model device, one per block."""
        F = self._torch.nn.functional
        dtype = next(self._model.parameters()).dtype
        with self._torch.inference_mode():
            feats = self._model.forward_intermediates(
                tensor.to(dtype),
                indices=list(self.extract_layers),
                norm=True,
                return_prefix_tokens=False,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        stored = {}
        for idx, feat in zip(self.extract_layers, feats):
            stored[idx] = F.normalize(feat.float(), dim=1)
        return stored

    def layer_maps(
        self,
        img: Image.Image,
        layers: Sequence[int | str] | None = None,
        *,
        cache_key: str | None = None,
    ) -> tuple[np.ndarray, ...]:
        """L2-normalised maps for ``layers``, each ``(H, W, C)`` on the patch grid."""
        names = resolve_dino_layers(layers) if layers is not None else self.extract_layers
        all_maps = self._all_layer_maps(img, cache_key=cache_key)
        missing = [n for n in names if n not in all_maps]
        if missing:
            raise ValueError(
                f"DinoEncoder was built with layers {self.extract_layers}, "
                f"cannot score {missing}"
            )
        return tuple(all_maps[name] for name in names)

    def change_map(
        self,
        clean: Image.Image,
        edit: Image.Image,
        *,
        cache_key: str | None = None,
        edit_cache_key: str | None = None,
        layers: Sequence[int | str] | None = None,
        layer_weights: Sequence[float] | None = None,
        shift_remove: float = 1.0,
        cosine_weight: float = 1.0,
        l2_weight: float = 0.0,
    ) -> np.ndarray:
        """Per-pixel DINO visual change at ``clean``'s resolution.

        Each selected block is scored with :func:`local_feature_distance`, upsampled to
        the crop, then mixed. ``shift_remove`` drops a uniform feature shift, so a global
        re-render does not register as change everywhere.
        """
        use_layers = resolve_dino_layers(layers) if layers is not None else self.extract_layers
        F = self._torch.nn.functional
        # Batched GPU forward when nothing is cached (the reblend path). Two separate
        # host round-trips plus a numpy distance is why nvtop showed 0% GPU.
        if cache_key is None and edit_cache_key is None:
            batch = self._torch.cat([self._prepare(clean), self._prepare(edit)], dim=0)
            feats = self._forward_batch(batch)
            maps = []
            width, height = clean.size
            for name in use_layers:
                grid = feats[name]
                dist = _local_feature_distance_torch(
                    grid[0].permute(1, 2, 0),
                    grid[1].permute(1, 2, 0),
                    shift_remove=shift_remove,
                    cosine_weight=cosine_weight,
                    l2_weight=l2_weight,
                )
                maps.append(
                    F.interpolate(
                        dist[None, None].float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                )
            stacked = self._torch.stack(maps, dim=0)
            if layer_weights is None or len(layer_weights) == 0:
                w = self._torch.ones(
                    stacked.shape[0], device=stacked.device, dtype=stacked.dtype
                )
            else:
                w = self._torch.as_tensor(
                    layer_weights, device=stacked.device, dtype=stacked.dtype
                )
                if not bool((w > 0).any()):
                    w = self._torch.ones_like(w)
            w = w / w.sum()
            return (w.view(-1, 1, 1) * stacked).sum(dim=0).detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        clean_tok = self.layer_maps(clean, use_layers, cache_key=cache_key)
        edit_tok = self.layer_maps(edit, use_layers, cache_key=edit_cache_key)
        maps = [
            upsample_change_map(
                local_feature_distance(
                    a,
                    b,
                    shift_remove=shift_remove,
                    cosine_weight=cosine_weight,
                    l2_weight=l2_weight,
                ),
                clean.size,
            )
            for a, b in zip(clean_tok, edit_tok)
        ]
        return combine_layer_maps(maps, layer_weights)


def _dino_pretrained(weights: str | None | bool) -> bool:
    """``None`` / ``none`` / ``false`` → random init (tests); anything else → timm weights."""
    if weights in (None, False, "none", "None", "false", "False"):
        return False
    return True


def build_dino_encoder(cfg: dict | None, *, layers: Sequence[int | str] | None = None) -> DinoEncoder:
    """Construct the encoder described by a ``masks.dinov3`` (or post-blend) config block."""
    cfg = cfg or {}
    return DinoEncoder(
        str(cfg.get("model", DEFAULT_DINO_MODEL)),
        device=str(cfg.get("device", "cuda:0")),
        cache_size=int(cfg.get("cache_size", 8)),
        pretrained=_dino_pretrained(cfg.get("weights", "default")),
        layers=layers if layers is not None else cfg.get("layers"),
    )


def dino_change_fn(
    encoder: DinoEncoder,
    cfg: dict | None = None,
    *,
    cache_key: str | None = None,
    edit_cache_key: str | None = None,
):
    """Bind ``encoder.change_map`` to a ``masks.dinov3`` block for :func:`estimate_mask`."""
    cfg = cfg or {}
    layers = resolve_dino_layers(cfg.get("layers"))
    weights = cfg.get("layer_weights")
    shift = float(cfg.get("shift_remove", 1.0))
    cosine_w = float(cfg.get("cosine_weight", 1.0))
    l2_w = float(cfg.get("l2_weight", 0.0))

    def fn(c, e, _enc=encoder):
        return _enc.change_map(
            c,
            e,
            cache_key=cache_key,
            edit_cache_key=edit_cache_key,
            layers=layers,
            layer_weights=weights,
            shift_remove=shift,
            cosine_weight=cosine_w,
            l2_weight=l2_w,
        )

    return fn


# --- ResNet-18 change map ---------------------------------------------------

# ImageNet-pretrained CNN feature maps. layer2 is 1/8 of the crop (128×128 at 1024) —
# denser than DINOv3's 1/16 patch grid. layer3 is 1/16, matching DINOv3-S/16. Default
# mix is the PatchCore CNN tap (layer2+layer3). Torchvision is imported lazily so
# CPU-only tests stay cheap.
RESNET_LAYERS: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
DEFAULT_RESNET_LAYERS: tuple[str, ...] = ("layer2", "layer3")
_RESNET_FACTOR = 32  # stem + 4 stages of stride 2


def resolve_resnet_layers(layers: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and freeze a layer tuple; empty / None → :data:`DEFAULT_RESNET_LAYERS`."""
    raw = tuple(str(n) for n in layers) if layers else DEFAULT_RESNET_LAYERS
    if not raw:
        raw = DEFAULT_RESNET_LAYERS
    unknown = [n for n in raw if n not in RESNET_LAYERS]
    if unknown:
        raise ValueError(
            f"unknown ResNet layer(s) {unknown}; expected a subset of {RESNET_LAYERS}"
        )
    return raw


class ResNetEncoder:
    """ImageNet ResNet feature maps → cosine-distance change map for a crop pair.

    One forward extracts ``layer1``–``layer4``. Callers pick a subset and mix with
    :func:`combine_layer_maps` after each layer is scored and upsampled to crop size
    (the layers do not share a grid, unlike ViT blocks). Clean-crop maps are cached by
    ``cache_key`` so re-scoring the same clean crop does not re-forward.
    """

    def __init__(
        self,
        model: str = "resnet18",
        *,
        device: str = "cuda:0",
        cache_size: int = 8,
        weights: str | None = "imagenet",
    ) -> None:
        import torch
        from torchvision.models import resnet18, resnet34, resnet50
        from torchvision.models.feature_extraction import create_feature_extractor

        builders = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
        if model not in builders:
            raise ValueError(f"unknown ResNet {model!r}; expected one of {tuple(builders)}")
        self.name = str(model)
        self.device = device
        self._torch = torch
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = int(cache_size)
        tv_weights = _resnet_weights(self.name, weights)
        backbone = builders[self.name](weights=tv_weights)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._extractor = create_feature_extractor(
            backbone.to(device), return_nodes={n: n for n in RESNET_LAYERS}
        ).eval()

    def _prepare(self, img: Image.Image):
        """ImageNet-normalised ``(1, 3, H, W)`` tensor, sides rounded up to a multiple of 32."""
        torch = self._torch
        rgb = img.convert("RGB")
        w, h = rgb.size
        tw = max(_RESNET_FACTOR, ((w + _RESNET_FACTOR - 1) // _RESNET_FACTOR) * _RESNET_FACTOR)
        th = max(_RESNET_FACTOR, ((h + _RESNET_FACTOR - 1) // _RESNET_FACTOR) * _RESNET_FACTOR)
        if (w, h) != (tw, th):
            rgb = rgb.resize((tw, th), _RESAMPLE)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return (
            torch.from_numpy((arr - _IMAGENET_MEAN) / _IMAGENET_STD)
            .permute(2, 0, 1)[None]
            .to(self.device)
        )

    def _all_layer_maps(self, img: Image.Image, cache_key: str | None = None) -> dict[str, np.ndarray]:
        """L2-normalised ``(H, W, C)`` maps for every ResNet stage, optionally memoised."""
        if cache_key is not None and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        tensor = self._prepare(img)
        with self._torch.no_grad():
            feats = self._extractor(tensor)
        stored: dict[str, np.ndarray] = {}
        for name, feat in feats.items():
            tok = feat[0].permute(1, 2, 0).float().cpu().numpy()
            stored[name] = tok / (np.linalg.norm(tok, axis=-1, keepdims=True) + 1e-8)
        if cache_key is not None:
            self._cache[cache_key] = stored
            while len(self._cache) > max(self._cache_size, 1):
                self._cache.popitem(last=False)
        return stored

    def layer_maps(
        self,
        img: Image.Image,
        layers: Sequence[str] = DEFAULT_RESNET_LAYERS,
        *,
        cache_key: str | None = None,
    ) -> tuple[np.ndarray, ...]:
        """L2-normalised maps for ``layers``, each ``(H, W, C)`` at that stage's stride."""
        names = resolve_resnet_layers(layers)
        all_maps = self._all_layer_maps(img, cache_key=cache_key)
        return tuple(all_maps[name] for name in names)

    def change_map(
        self,
        clean: Image.Image,
        edit: Image.Image,
        *,
        cache_key: str | None = None,
        edit_cache_key: str | None = None,
        layers: Sequence[str] | None = None,
        layer_weights: Sequence[float] | None = None,
        shift_remove: float = 1.0,
        cosine_weight: float = 1.0,
        l2_weight: float = 0.0,
    ) -> np.ndarray:
        """Per-pixel ResNet visual change at ``clean``'s resolution.

        Each selected stage is scored with :func:`local_feature_distance`, upsampled to
        the crop, then mixed. ``shift_remove`` drops a uniform feature shift, so a global
        re-render does not register as change everywhere.
        """
        use_layers = resolve_resnet_layers(layers)
        clean_tok = self.layer_maps(clean, use_layers, cache_key=cache_key)
        edit_tok = self.layer_maps(edit, use_layers, cache_key=edit_cache_key)
        maps = [
            upsample_change_map(
                local_feature_distance(
                    a,
                    b,
                    shift_remove=shift_remove,
                    cosine_weight=cosine_weight,
                    l2_weight=l2_weight,
                ),
                clean.size,
            )
            for a, b in zip(clean_tok, edit_tok)
        ]
        return combine_layer_maps(maps, layer_weights)


def _resnet_weights(model: str, weights: str | None):
    """``imagenet`` → torchvision ImageNet1K_V1; ``None`` / ``none`` → random init (tests)."""
    if weights in (None, False, "none", "None"):
        return None
    if str(weights).lower() not in ("imagenet", "default", "imagenet1k_v1"):
        raise ValueError(
            f"unknown ResNet weights {weights!r}; expected 'imagenet' or None"
        )
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

    table = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
    }
    return table[model]


def build_resnet_encoder(cfg: dict | None) -> ResNetEncoder:
    """Construct the encoder described by a ``masks.resnet`` config block."""
    cfg = cfg or {}
    return ResNetEncoder(
        str(cfg.get("model", "resnet18")),
        device=str(cfg.get("device", "cuda:0")),
        cache_size=int(cfg.get("cache_size", 8)),
        weights=cfg.get("weights", "imagenet"),
    )


def resnet_change_fn(
    encoder: ResNetEncoder,
    cfg: dict | None = None,
    *,
    cache_key: str | None = None,
    edit_cache_key: str | None = None,
):
    """Bind ``encoder.change_map`` to a ``masks.resnet`` block for :func:`estimate_mask`."""
    cfg = cfg or {}
    layers = resolve_resnet_layers(cfg.get("layers"))
    weights = cfg.get("layer_weights")
    shift = float(cfg.get("shift_remove", 1.0))
    cosine_w = float(cfg.get("cosine_weight", 1.0))
    l2_w = float(cfg.get("l2_weight", 0.0))

    def fn(c, e, _enc=encoder):
        return _enc.change_map(
            c,
            e,
            cache_key=cache_key,
            edit_cache_key=edit_cache_key,
            layers=layers,
            layer_weights=weights,
            shift_remove=shift,
            cosine_weight=cosine_w,
            l2_weight=l2_w,
        )

    return fn


def load_change_fns(masks: dict | None) -> tuple:
    """Paste-seed and optional post-blend change maps from ``masks.encoder``.

    One backbone is loaded; the two closures differ only in which layers /
    ``shift_remove`` / distance mix they read from their config block.
    ``post_semantic_fn`` is ``None`` when ``masks.post_blend.enabled`` is false.
    """
    masks = masks or {}
    kind = resolve_encoder_kind(masks)
    device = str(masks.get("device") or "cuda:0")
    post_cfg = dict(masks.get("post_blend") or {})
    if not post_cfg.get("device"):
        post_cfg["device"] = device
    post_on = bool(post_cfg.get("enabled", False))

    if kind == ENCODER_DINO:
        paste_cfg = dict(masks.get("dinov3") or {})
        if not paste_cfg.get("device"):
            paste_cfg["device"] = device
        paste_layers = resolve_dino_layers(paste_cfg.get("layers"))
        post_layers = resolve_dino_layers(post_cfg.get("layers")) if post_on else ()
        encoder = build_dino_encoder(
            paste_cfg, layers=_unique_ints((*paste_layers, *post_layers))
        )
        print(
            f"[dinov3] {encoder.name} on {next(encoder._model.parameters()).device} "
            f"layers={paste_layers}",
            flush=True,
        )
        post_fn = None
        if post_on:
            post_fn = dino_change_fn(encoder, post_cfg)
            print(f"[dinov3] post-blend GT layers={post_layers}", flush=True)
        return dino_change_fn(encoder, paste_cfg), post_fn

    paste_cfg = dict(masks.get("resnet") or {})
    if not paste_cfg.get("device"):
        paste_cfg["device"] = device
    encoder = build_resnet_encoder(paste_cfg)
    print(
        f"[resnet] {encoder.name} on {next(encoder._extractor.parameters()).device} "
        f"layers={tuple(paste_cfg.get('layers') or DEFAULT_RESNET_LAYERS)}",
        flush=True,
    )
    post_fn = None
    if post_on:
        post_fn = resnet_change_fn(encoder, post_cfg)
        print(
            f"[resnet] post-blend GT layers="
            f"{tuple(post_cfg.get('layers') or DEFAULT_RESNET_LAYERS)}",
            flush=True,
        )
    return resnet_change_fn(encoder, paste_cfg), post_fn
