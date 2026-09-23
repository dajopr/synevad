"""Per-category overlays for generation / blending / masking config.

A YAML keyed by MVTec category, each block a *partial* of the standalone config.
Deep-merged on top of the base YAML so unlisted knobs stay at their defaults.

Two equivalent spellings for mask sub-blocks::

    metal_nut:
      post_blend:
        min_change: 0.15

    metal_nut:
      masks:
        post_blend:
          min_change: 0.15

Top-level keys under a category must be a config section (``generation``,
``blending``, ``cropping``, ``masks``, ``scoring``) or a mask-sub-block shorthand
(``post_blend``, ``resnet``, ``dinov3``, ``extent``, ``encoder``). Anything else fails loudly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

SECTIONS: frozenset[str] = frozenset(
    {"generation", "blending", "cropping", "masks", "scoring", "negatives"}
)
# Written at the category root, these merge into ``masks.<name>``.
MASK_SHORTHAND: frozenset[str] = frozenset(
    {"post_blend", "resnet", "dinov3", "extent", "encoder"}
)


def _as_omegaconf(node: Any) -> DictConfig | ListConfig:
    if isinstance(node, (DictConfig, ListConfig)):
        return node
    return OmegaConf.create(node)


def normalize_override_block(block: Any, *, category: str) -> DictConfig:
    """Map one category's overlay onto top-level standalone-config keys."""
    if block is None:
        return OmegaConf.create({})
    node = _as_omegaconf(block)
    if not OmegaConf.is_dict(node):
        raise ValueError(
            f"override for {category!r} must be a mapping, got {type(block).__name__}"
        )
    merged = OmegaConf.create({})
    unknown: list[str] = []
    for key, value in node.items():
        name = str(key)
        if name in SECTIONS:
            piece = OmegaConf.create({name: value})
        elif name in MASK_SHORTHAND:
            piece = OmegaConf.create({"masks": {name: value}})
        else:
            unknown.append(name)
            continue
        merged = OmegaConf.merge(merged, piece)
    if unknown:
        raise ValueError(
            f"unknown override key(s) for {category!r}: {', '.join(sorted(unknown))}. "
            "Nest under generation/, blending/, cropping/, masks/, scoring/, or negatives/; "
            "post_blend / resnet / dinov3 / extent / encoder are shorthand for masks.<name>"
        )
    return merged


def apply_category_overrides(
    cfg: DictConfig | Mapping,
    category: str | None,
    overrides: Mapping[str, Any] | None,
) -> DictConfig:
    """Deep-merge ``overrides[category]`` onto ``cfg``. Unlisted categories are a no-op."""
    cfg = _as_omegaconf(cfg)
    if not category or not overrides:
        return cfg
    block = overrides.get(str(category))
    if block is None:
        return cfg
    overlay = normalize_override_block(block, category=str(category))
    if not OmegaConf.to_container(overlay, resolve=True):
        return cfg
    return OmegaConf.merge(cfg, overlay)


def flatten_overlay(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Dotted ``(key, value)`` pairs of an overlay, for the log line."""
    if node is None:
        return []
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)
    if isinstance(node, Mapping):
        out: list[tuple[str, Any]] = []
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_overlay(value, path))
        return out
    return [(prefix, node)]


def format_applied(category: str, overlay: DictConfig | Mapping) -> str:
    items = flatten_overlay(overlay)
    if not items:
        return f"[overrides] {category}: (none)"
    body = ", ".join(f"{k}={v}" for k, v in items)
    return f"[overrides] {category}: {body}"


def overlay_for(category: str, overrides: Mapping[str, Any] | None) -> DictConfig:
    """The normalized overlay that would be merged for ``category`` (empty if none)."""
    if not category or not overrides:
        return OmegaConf.create({})
    block = overrides.get(str(category))
    if block is None:
        return OmegaConf.create({})
    return normalize_override_block(block, category=str(category))


def resolve_overrides_file(spec: str, *, config_path: Path | str | None = None) -> Path:
    """Locate an overrides YAML: as given, then next to the generation config."""
    p = Path(spec).expanduser()
    candidates = [p]
    if not p.is_absolute() and config_path is not None:
        parent = Path(config_path).expanduser().resolve().parent
        candidates.append(parent / spec)
        candidates.append(parent / Path(spec).name)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise SystemExit(f"overrides file not found: {spec}")


def _overrides_node_to_conf(raw: Any, *, config_path: Path | str | None) -> Any:
    """Interpret ``cfg.overrides`` as a file path or an inline category mapping."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return OmegaConf.load(str(resolve_overrides_file(raw, config_path=config_path)))
    container = OmegaConf.to_container(_as_omegaconf(raw), resolve=True)
    if container is None:
        return None
    if isinstance(container, str):
        return OmegaConf.load(str(resolve_overrides_file(container, config_path=config_path)))
    if isinstance(container, dict):
        return _as_omegaconf(container)
    raise SystemExit("config.overrides must be a YAML path or a mapping of category → overlay")


def load_override_map(
    cfg: Any = None,
    *,
    config_path: Path | str | None = None,
    cli_path: str | None = None,
) -> dict[str, Any]:
    """Category → overlay block, from ``cfg.overrides`` and/or ``--overrides``.

    ``cfg.overrides`` may be a file path or an inline mapping. ``--overrides`` is
    always a file and is merged on top of whatever the config supplied.
    """
    pieces: list[Any] = []
    raw = None if cfg is None else cfg.get("overrides")
    loaded = _overrides_node_to_conf(raw, config_path=config_path)
    if loaded is not None:
        pieces.append(loaded)
    if cli_path:
        pieces.append(
            OmegaConf.load(str(resolve_overrides_file(cli_path, config_path=config_path)))
        )
    if not pieces:
        return {}
    merged = pieces[0] if len(pieces) == 1 else OmegaConf.merge(*pieces)
    out = OmegaConf.to_container(merged, resolve=True)
    return dict(out) if isinstance(out, dict) else {}


def apply_loaded_overrides(
    cfg: DictConfig,
    category: str | None,
    overrides: Mapping[str, Any] | None,
) -> tuple[DictConfig, DictConfig]:
    """Apply the overlay and return ``(merged_cfg, overlay_that_was_applied)``."""
    overlay = overlay_for(str(category or ""), overrides)
    if not OmegaConf.to_container(overlay, resolve=True):
        return cfg, overlay
    return OmegaConf.merge(cfg, overlay), overlay
