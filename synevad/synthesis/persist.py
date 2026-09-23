"""On-disk layout for one standalone generation candidate.

Every candidate writes the clean crop, the unblended FLUX edit, the two composites, the
four masks, and RGB overlays of the binary GT on both composites. The raw edit is what
lets a later pass reblend with a new mask without re-running FLUX.

Everything here is recoverable offline from ``crop_original/`` + ``crop_edited/``, which is
what lets a mask be re-derived and re-composited without re-running FLUX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from synevad.synthesis.masks import write_masks

EDITED_DIRNAME = "crop_edited"
# Negative controls: the pre-chosen region each inpaint was confined to (binary, crop res).
REGION_DIRNAME = "region"


@dataclass(frozen=True)
class ArtifactDirs:
    """Subdirectories under a run's ``out_dir``."""

    clean: Path
    edited: Path
    blended: Path
    composited: Path
    mask_crop: Path
    mask_soft_crop: Path
    mask_full: Path
    mask_soft_full: Path
    overlay_crop: Path
    overlay_full: Path

    def mkdir(self) -> None:
        for path in (
            self.clean,
            self.edited,
            self.blended,
            self.composited,
            self.mask_crop,
            self.mask_soft_crop,
            self.mask_full,
            self.mask_soft_full,
            self.overlay_crop,
            self.overlay_full,
        ):
            path.mkdir(parents=True, exist_ok=True)


def artifact_dirs(out_dir: Path) -> ArtifactDirs:
    """The standard layout under ``out_dir`` (directories are not created)."""
    root = Path(out_dir)
    return ArtifactDirs(
        clean=root / "crop_original",
        edited=root / EDITED_DIRNAME,
        blended=root / "crop_blended",
        composited=root / "composited",
        mask_crop=root / "mask_crop",
        mask_soft_crop=root / "mask_soft_crop",
        mask_full=root / "mask_full",
        mask_soft_full=root / "mask_soft_full",
        overlay_crop=root / "mask_overlay_crop",
        overlay_full=root / "mask_overlay_full",
    )


def raw_edit_path(dirs: ArtifactDirs, name: str) -> Path:
    """Filename for the unblended FLUX crop of candidate ``name``."""
    return dirs.edited / f"{name}.png"


def region_path(dirs: ArtifactDirs, name: str) -> Path:
    """Filename for the region mask of negative-control candidate ``name``."""
    return dirs.clean.parent / REGION_DIRNAME / f"{name}.png"


def write_region(dirs: ArtifactDirs, name: str, image: Image.Image) -> Path:
    """Persist a negative's region *before* the inpaint, write-then-rename like the raw edit.

    The blend reads it back instead of re-deriving it, so a later change to the region
    sampler can never make a stored inpaint and its composite alpha disagree.
    """
    path = region_path(dirs, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(path)
    return path


def write_raw_edit(dirs: ArtifactDirs, name: str, image: Image.Image) -> Path:
    """Persist the unblended editor output; call this *before* blending.

    Write-then-rename so a polling mask worker never opens a half-written PNG.
    """
    dirs.edited.mkdir(parents=True, exist_ok=True)
    path = raw_edit_path(dirs, name)
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(path)
    return path


def persist_composite(
    dirs: ArtifactDirs,
    *,
    name: str,
    roi_id: str,
    source_path: Path,
    box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    clean_crop: Image.Image,
    full: Image.Image,
    patch: Image.Image,
    mask,
    raw_path: Path,
) -> dict[str, str]:
    """Write original crop, composites, masks, and mask overlays; return resolved path fields.

    The raw edit is assumed to already be on disk at ``raw_path`` (written before the
    blend). The clean crop is written once per ROI.
    """
    dirs.mkdir()
    clean_path = dirs.clean / f"{roi_id}__clean.png"
    if not clean_path.exists():
        clean_crop.save(clean_path)
    patch_path = dirs.blended / f"{name}.png"
    full_path = dirs.composited / f"{name}_full.png"
    patch.save(patch_path)
    full.save(full_path)
    mask_paths = write_masks(
        mask,
        box,
        source_size,
        mask_crop_dir=dirs.mask_crop,
        mask_soft_crop_dir=dirs.mask_soft_crop,
        mask_full_dir=dirs.mask_full,
        mask_soft_full_dir=dirs.mask_soft_full,
        overlay_crop_dir=dirs.overlay_crop,
        overlay_full_dir=dirs.overlay_full,
        patch=patch,
        full=full,
        name=name,
    )
    return {
        "source_path": str(Path(source_path).resolve()),
        "clean_crop_path": str(clean_path.resolve()),
        "raw_edit_path": str(Path(raw_path).resolve()),
        "blended_patch_path": str(patch_path.resolve()),
        "full_blended_path": str(full_path.resolve()),
        **mask_paths,
    }
