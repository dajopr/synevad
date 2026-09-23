from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


def get_default_transforms(size: tuple[int, int]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(size),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class Sample:
    image: Tensor
    mask: Tensor
    label: int
    class_name: str
    image_path: str
    # None for samples without a ground-truth mask (defect-free images).
    mask_path: str | None

    scorer_result: str = "real"
    severity: str = "real"
    # The mask estimator's verdict for a generated defect (`synevad.synthesis.masks`): "ok" means a
    # usable mask was written, anything else names why it was not. "real" for photographed
    # defects, whose mask is ground truth, and "none" for defect-free frames.
    mask_status: str = "real"
    # EditReward margin for a generated defect; None for real images, defect-free frames
    # and rows the judge never scored.
    reward_score: float | None = None
    # Share of the frame the mask covers, for a generated defect; None for real images,
    # defect-free frames and rows whose mask was never measured. Zero is a real
    # measurement (an empty mask), which is what the synthetic arms' gate drops.
    defect_area_frac: float | None = None


@dataclass
class Batch:
    # None once the images have been embedded: `EvalFeatureCache` keeps a batch's
    # metadata alive across every config that shares a feature stage, and the decoded
    # tensor is the bulk of it.
    image: Tensor | None
    mask: Tensor
    label: Tensor
    class_name: list[str]
    image_paths: list[str]
    mask_paths: list[str | None]

    scorer_result: list[str]
    severity: list[str]
    mask_status: list[str]
    reward_score: list[float | None]
    defect_area_frac: list[float | None]

    score: Tensor | None = None
    anomaly_map: Tensor | None = None

    def with_predictions(self, score: Tensor, anomaly_map: list[np.ndarray]) -> "Batch":
        """A copy carrying predictions, leaving the cached source batch untouched.

        A copy rather than a mutation because the source is shared: writing predictions
        into it would leave one config's scores hanging off the object the next 35
        configs reuse.
        """
        return replace(
            self, score=score, anomaly_map=torch.tensor(np.stack(anomaly_map))
        )


def train_collate_fn(batch: list[Sample]) -> Tensor:
    image = torch.stack([s.image for s in batch])
    return image


def collate_fn(batch: list[Sample]) -> Batch:
    image = torch.stack([s.image for s in batch])
    mask = torch.stack([s.mask for s in batch])
    label = torch.tensor([s.label for s in batch], dtype=torch.long)
    class_names = [s.class_name for s in batch]
    image_paths = [s.image_path for s in batch]
    mask_paths = [s.mask_path for s in batch]
    scorer_results = [s.scorer_result for s in batch]
    severities = [s.severity for s in batch]
    mask_statuses = [s.mask_status for s in batch]
    reward_scores = [s.reward_score for s in batch]
    defect_area_fracs = [s.defect_area_frac for s in batch]

    return Batch(
        image=image,
        mask=mask,
        label=label,
        class_name=class_names,
        image_paths=image_paths,
        mask_paths=mask_paths,
        severity=severities,
        scorer_result=scorer_results,
        mask_status=mask_statuses,
        reward_score=reward_scores,
        defect_area_frac=defect_area_fracs,
    )


class ImageRecord(Protocol):
    image_path: Path
    mask_path: Path | None
    label: int
    class_name: str

    severity: str
    scorer_result: str
    mask_status: str
    reward_score: float | None
    defect_area_frac: float | None


class ImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: T.Compose,
    ) -> None:
        super().__init__()
        self.records = records
        self.transform = transform

    def __getitem__(self, idx: int) -> Sample:
        image_path = self.records[idx].image_path
        image_np = np.array(Image.open(image_path).convert("RGB"))

        mask_path = self.records[idx].mask_path
        if mask_path is None:
            mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
        else:
            mask_np = (np.asarray(Image.open(mask_path)) > 0).astype(np.uint8)

        image: tv_tensors.Image = self.transform(
            tv_tensors.Image(image_np.transpose(2, 0, 1))
        )
        mask: tv_tensors.Mask = self.transform(tv_tensors.Mask(mask_np)).unsqueeze(0)

        sample = Sample(
            image=image,
            mask=mask,
            label=self.records[idx].label,
            class_name=self.records[idx].class_name,
            image_path=str(image_path),
            mask_path=None if mask_path is None else str(mask_path),
            scorer_result=self.records[idx].scorer_result,
            severity=self.records[idx].severity,
            mask_status=self.records[idx].mask_status,
            reward_score=self.records[idx].reward_score,
            defect_area_frac=self.records[idx].defect_area_frac,
        )

        return sample

    def __len__(self) -> int:
        return len(self.records)
