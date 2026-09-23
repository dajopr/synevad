"""DRAEM-style Perlin + DTD overlays as a deterministic, persistable eval augmenter.

Official DRAEM (Zavrtanik et al., ICCV 2021) overlays a thresholded Perlin mask of a
DTD texture onto a *training* image every epoch, and half the time returns the clean
frame unchanged. Synevad cannot use that loop: synthetic images are an eval arm, so the
set has to be identical for every config in a sweep or choice regret is noise.

This module is the overlay itself — one draw, seeded, no 50% skip — so a generator can
write a frozen corpus. The blend is the official loader's formula, not the paper's
typesetting of it:

    I_a = I * (1 - M) + (1 - β) * (A * M) + β * (I * M)

with ``β ~ U(0, 0.8)`` as in ``VitjanZ/DRAEM/data_loader.py`` (the paper writes
``[0.1, 1.0]``). Texture jitter is a three-op sample from the same family the loader
uses, implemented in numpy/PIL so this package does not grow an ``imgaug`` dependency.

Perlin is generated at a size divisible by the chosen scale and cropped back to the
source resolution, so the mask is measured on the native composite the way FLUX blend
measures the native crop, rather than being locked to DRAEM's 256 training size.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

MIN_PERLIN_SCALE: int = 0
MAX_PERLIN_SCALE: int = 6  # exclusive: 2^{0..5} = 1, 2, 4, 8, 16, 32
PERLIN_THRESHOLD: float = 0.5
BETA_MAX: float = 0.8
N_TEXTURE_AUGMENTS: int = 3
MAX_MASK_ATTEMPTS: int = 8

CLASS_NAME: str = "draem"
SEVERITY: str = "draem"


@dataclass(frozen=True)
class DraemMeta:
    """The draw that produced one overlay, for the generation manifest."""

    texture_path: str | None
    perlin_scalex: int
    perlin_scaley: int
    beta: float
    n_mask: int
    attempts: int


def overlay_rng(seed: int, category: str, source_id: str, overlay_i: int) -> Generator:
    """A generator unique to ``(seed, category, source, overlay)``, stable across processes.

    ``hash()`` is per-process; a SHA-256 of the identity is not, so a resume on another
    worker reproduces the same overlay.
    """
    payload = f"{int(seed)}\0{category}\0{source_id}\0{int(overlay_i)}".encode()
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(derived)


def list_dtd_textures(root: str | Path) -> list[Path]:
    """Image files under a DTD-style tree (``images/<class>/*.jpg``), or any nested images."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DTD root is not a directory: {root}")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"no texture images under {root}")
    return paths


def rand_perlin_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    rng: Generator,
    fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
) -> np.ndarray:
    """The official ``rand_perlin_2d_np``, driven by ``rng`` instead of global ``np.random``.

    ``shape`` must be divisible by ``res`` on each axis — :func:`binary_perlin_mask`
    pads before calling this.
    """
    height, width = shape
    res_y, res_x = res
    if height % res_y or width % res_x:
        raise ValueError(
            f"Perlin shape {shape} is not divisible by resolution {res}"
        )

    delta = (res_y / height, res_x / width)
    d = (height // res_y, width // res_x)
    grid = np.mgrid[0:res_y:delta[0], 0:res_x:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * rng.random((res_y + 1, res_x + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(slice1, slice2):
        return np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]],
                d[0],
                axis=0,
            ),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        return (
            np.stack(
                (
                    grid[:height, :width, 0] + shift[0],
                    grid[:height, :width, 1] + shift[1],
                ),
                axis=-1,
            )
            * grad[:height, :width]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:height, :width])
    return math.sqrt(2) * _lerp(_lerp(n00, n10, t[..., 0]), _lerp(n01, n11, t[..., 0]), t[..., 1])


def binary_perlin_mask(
    height: int,
    width: int,
    rng: Generator,
    *,
    min_scale: int = MIN_PERLIN_SCALE,
    max_scale: int = MAX_PERLIN_SCALE,
    threshold: float = PERLIN_THRESHOLD,
) -> tuple[np.ndarray, int, int]:
    """Thresholded Perlin mask at ``(height, width)``, plus the scales that made it.

    Pads up to a multiple of each scale so the official generator's integer tiling
    holds, rotates the field the way the loader's ``Affine(rotate=(-90, 90))`` does,
    then crops back.
    """
    scalex = 2 ** int(rng.integers(min_scale, max_scale))
    scaley = 2 ** int(rng.integers(min_scale, max_scale))
    h_pad = int(math.ceil(height / scaley) * scaley)
    w_pad = int(math.ceil(width / scalex) * scalex)
    noise = rand_perlin_2d((h_pad, w_pad), (scaley, scalex), rng)
    angle = float(rng.uniform(-90.0, 90.0))
    noise = ndimage.rotate(noise, angle, reshape=False, order=1, mode="reflect")
    noise = noise[:height, :width]
    mask = (noise > threshold).astype(np.float32)
    return mask, scalex, scaley


def blend(
    image: np.ndarray,
    texture: np.ndarray,
    mask: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Official DRAEM overlay. ``image``/``texture`` uint8 RGB, ``mask`` in ``{0, 1}``."""
    image_f = image.astype(np.float32) / 255.0
    texture_f = texture.astype(np.float32) / 255.0
    mask_f = np.asarray(mask, dtype=np.float32)
    if mask_f.ndim == 2:
        mask_f = mask_f[..., None]
    img_thr = texture_f * mask_f
    augmented = image_f * (1.0 - mask_f) + (1.0 - beta) * img_thr + beta * image_f * mask_f
    augmented = mask_f * augmented + (1.0 - mask_f) * image_f
    return np.clip(np.round(augmented * 255.0), 0, 255).astype(np.uint8)


def augment_texture(texture: np.ndarray, rng: Generator) -> np.ndarray:
    """Three random ops from the official loader's augmenter list, minus ``imgaug``."""
    ops = (
        _aug_gamma,
        _aug_brightness,
        _aug_sharpness,
        _aug_hue_sat,
        _aug_solarize,
        _aug_posterize,
        _aug_invert,
        _aug_autocontrast,
        _aug_equalize,
        _aug_rotate,
    )
    chosen = rng.choice(len(ops), size=N_TEXTURE_AUGMENTS, replace=False)
    out = texture
    for index in chosen:
        out = ops[int(index)](out, rng)
    return out


def augment(
    image: np.ndarray,
    *,
    rng: Generator,
    texture: np.ndarray | None = None,
    dtd_paths: Sequence[Path] | None = None,
    threshold: float = PERLIN_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, DraemMeta]:
    """One DRAEM overlay. Always applies; empty Perlin is a possible (rare) outcome.

    Supply either ``texture`` (tests) or ``dtd_paths`` (generation). The mask is uint8
    ``{0, 255}`` at the source resolution.
    """
    image = _as_rgb(image)
    height, width = image.shape[:2]
    texture_path: str | None = None
    if texture is None:
        if not dtd_paths:
            raise ValueError("augment needs a texture array or a non-empty dtd_paths")
        path = Path(dtd_paths[int(rng.integers(0, len(dtd_paths)))])
        texture_path = str(path)
        texture = _load_rgb(path, (width, height))
    else:
        texture = _resize_rgb(_as_rgb(texture), (width, height))

    texture = augment_texture(texture, rng)
    mask, scalex, scaley = binary_perlin_mask(
        height, width, rng, threshold=threshold
    )
    beta = float(rng.random() * BETA_MAX)
    blended = blend(image, texture, mask, beta)
    binary = (mask > 0.5).astype(np.uint8) * 255
    return blended, binary, DraemMeta(
        texture_path=texture_path,
        perlin_scalex=scalex,
        perlin_scaley=scaley,
        beta=beta,
        n_mask=int((binary > 0).sum()),
        attempts=1,
    )


def augment_nonzero(
    image: np.ndarray,
    *,
    rng: Generator,
    texture: np.ndarray | None = None,
    dtd_paths: Sequence[Path] | None = None,
    max_attempts: int = MAX_MASK_ATTEMPTS,
) -> tuple[np.ndarray, np.ndarray, DraemMeta]:
    """:func:`augment` retried until the mask has pixels, or ``max_attempts`` is spent.

    Official DRAEM keeps empty Perlin as ``has_anomaly=0`` training noise. Synevad's gate
    drops those rows, so burning a few extra draws to get a real defect is the eval-set
    reading of the same generator. The last draw is returned even if it is still empty.
    """
    last: tuple[np.ndarray, np.ndarray, DraemMeta] | None = None
    for attempt in range(1, max_attempts + 1):
        blended, binary, meta = augment(
            image, rng=rng, texture=texture, dtd_paths=dtd_paths
        )
        last = (
            blended,
            binary,
            DraemMeta(
                texture_path=meta.texture_path,
                perlin_scalex=meta.perlin_scalex,
                perlin_scaley=meta.perlin_scaley,
                beta=meta.beta,
                n_mask=meta.n_mask,
                attempts=attempt,
            ),
        )
        if meta.n_mask > 0:
            return last
    assert last is not None
    return last


def _lerp(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (y - x) * w + x


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _resize_rgb(image: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    width, height = size_xy
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
    )


def _load_rgb(path: Path, size_xy: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        return _resize_rgb(_as_rgb(np.asarray(im.convert("RGB"))), size_xy)


def _aug_gamma(image: np.ndarray, rng: Generator) -> np.ndarray:
    gamma = rng.uniform(0.5, 2.0, size=3).astype(np.float32)
    x = image.astype(np.float32) / 255.0
    return np.clip((x ** gamma) * 255.0, 0, 255).astype(np.uint8)


def _aug_brightness(image: np.ndarray, rng: Generator) -> np.ndarray:
    mul = float(rng.uniform(0.8, 1.2))
    add = float(rng.uniform(-30.0, 30.0))
    return np.clip(image.astype(np.float32) * mul + add, 0, 255).astype(np.uint8)


def _aug_sharpness(image: np.ndarray, rng: Generator) -> np.ndarray:
    factor = float(rng.uniform(0.0, 2.0))
    return np.asarray(ImageEnhance.Sharpness(Image.fromarray(image)).enhance(factor))


def _aug_hue_sat(image: np.ndarray, rng: Generator) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(image).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-50, 51))) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.integers(-50, 51)), 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"))


def _aug_solarize(image: np.ndarray, rng: Generator) -> np.ndarray:
    thresh = int(rng.integers(32, 129))
    return np.asarray(ImageOps.solarize(Image.fromarray(image), thresh))


def _aug_posterize(image: np.ndarray, rng: Generator) -> np.ndarray:
    bits = int(rng.integers(1, 8))
    return np.asarray(ImageOps.posterize(Image.fromarray(image), bits))


def _aug_invert(image: np.ndarray, rng: Generator) -> np.ndarray:
    return 255 - image


def _aug_autocontrast(image: np.ndarray, rng: Generator) -> np.ndarray:
    return np.asarray(ImageOps.autocontrast(Image.fromarray(image)))


def _aug_equalize(image: np.ndarray, rng: Generator) -> np.ndarray:
    return np.asarray(ImageOps.equalize(Image.fromarray(image)))


def _aug_rotate(image: np.ndarray, rng: Generator) -> np.ndarray:
    angle = float(rng.uniform(-45.0, 45.0))
    return np.asarray(
        Image.fromarray(image).rotate(
            angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)
        )
    )
