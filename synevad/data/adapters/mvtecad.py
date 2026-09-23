from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torchvision.transforms.v2 as T

from ..dataset import ImageDataset
from ..split_pools import train_pool_for_seed


@dataclass
class MVTecSample:
    image_path: Path
    mask_path: Path | None
    label: int
    class_name: str

    severity: str = "real"
    scorer_result: str = "real"
    # Photographed defects come with a hand-drawn ground-truth mask, so there is no
    # estimator verdict to record — the synthetic arm's mask gate does not apply here.
    mask_status: str = "real"
    # Real defects are not generated, so no editor prompt was ever judged for them and
    # nothing measured their mask area — the gate that reads it applies to synthetic sets.
    reward_score: float | None = None
    defect_area_frac: float | None = None


def make_mvtecad_dataset(
    root: str,
    category: str,
    transform: T.Compose,
    split: str = "val",
    num_train_samples: int = -1,
    seed: int | None = None,
):
    # MVTec AD only ships train/ and test/ splits; treat "val" as "test".
    cat_dir = Path(root) / category
    split_dir = cat_dir / ("test" if split == "val" else split)
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"no {split} split at {split_dir}. For a pooled split, build "
            "data/synevad/MVTec_pools (python scripts/make_split_pools.py) and point "
            "MVTEC_PATH at it."
        )

    records: list[MVTecSample] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        label = 0 if class_name == "good" else 1
        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = (
                cat_dir / "ground_truth" / class_name / f"{image_path.stem}_mask.png"
            )
            if not mask_path.exists():
                mask_path = None

            records.append(
                MVTecSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    class_name=class_name,
                    label=label,
                )
            )
    # The pooled split keeps only a few frames per seed in train/good (all the sweep ever
    # draws) so the rest can carry synthetic defects. The pools are disjoint, so without
    # this every seed would draw from the union and the seeds would overlap.
    pool = train_pool_for_seed(cat_dir, seed) if split == "train" else None
    if pool is not None:
        records = [record for record in records if record.image_path.name in pool]
    if num_train_samples > len(records):
        hint = (
            f" for seed {seed}. Rebuild the pooled split with "
            f"--train-frames {num_train_samples}"
            if pool is not None
            else ""
        )
        raise ValueError(
            f"num_train_samples={num_train_samples} but {category} offers only "
            f"{len(records)} {split} frames{hint}"
        )
    if num_train_samples > 0:
        records = np.random.choice(records, size=num_train_samples, replace=False)
    return ImageDataset(records, transform)
