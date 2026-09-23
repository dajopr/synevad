import json
import re
import zlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from torchvision.transforms import v2 as T

from .dataset import ImageDataset

OK_CLASS_NAMES = ["good"]

# `class_name` given to negative controls (gated negatives runs): defect-free like "good", but
# kept apart from the real defect-free frames so scores.csv can tell the two populations apart.
NEGATIVE_CLASS_NAME = "negative"


class ScorerResult(StrEnum):
    ALL = "all"
    ACCEPT = "accept"
    REJECT = "reject"
    NONE = "none"


class Severity(StrEnum):
    ALL = "all"
    MINIMAL = "minimal"
    SLIGHT = "slight"
    MODERATE = "moderate"
    SEVERE = "severe"
    NONE = "none"


@dataclass
class SynevadSample:
    image_path: Path
    mask_path: Path | None
    label: int
    severity: str
    scorer_result: str
    class_name: str
    # The mask estimator's verdict (`synevad.synthesis.masks.MaskResult.status`) as the manifest
    # recorded it — see `mask_status_value`. "ok" is the only value that means a usable
    # mask was written; every other row still has a mask file on disk, and is meant to be
    # excluded by this column rather than by a missing path.
    mask_status: str = "unknown"
    # The EditReward margin the gate decided on. None when the row carries none — see
    # `reward_margin`.
    reward_score: float | None = None
    # Share of the frame the mask covers, from generation. None when the row was never
    # scored — see `defect_area`. Zero is a real measurement (an empty mask), so it is
    # not the missing sentinel; the eval gate turns on exactly that distinction.
    defect_area_frac: float | None = None


def scorer_verdict(entry: dict) -> str | None:
    """EditReward verdict (``accept`` / ``reject``), or None if the row is still unscored.

    New manifests store this as ``scorer``. Older HITL corpora used ``vlm_decision``;
    this reads ``scorer`` when the key is present, otherwise the old column.
    """
    value = entry["scorer"] if "scorer" in entry else entry.get("vlm_decision")
    if value is None:
        return None
    return str(value).lower()


def reward_margin(entry: dict) -> float | None:
    """The row's EditReward margin (`reward_score`), or None when it never met the judge.

    Written by `scripts/score_manifests.py` and by the generator's inline scoring pass.
    Rows generated with `scoring.enabled: false`, and manifests older than the field, have
    it missing or null — which means "unscored", not zero, so it stays None all the way
    into `scores.csv` rather than becoming a margin the judge never gave.
    """
    value = entry.get("reward_score")
    return None if value is None else float(value)


def mask_status_value(entry: dict) -> str:
    """The row's `mask_status`, or "unknown" when the manifest predates the column.

    Written at blend time by `synevad.synthesis.masks.MaskResult.manifest_fields` (and by
    a re-masking pass). "unknown" rather than "ok": a manifest that never recorded a
    verdict cannot claim one, and the metric gate that reads this column should drop such
    a row rather than admit it as measured.
    """
    value = entry.get("mask_status")
    return "unknown" if value is None else str(value)


def defect_area(entry: dict) -> float | None:
    """The row's ``defect_area_frac``, or None when it was never measured.

    Written at blend time (`synevad.synthesis.standalone_run`). Missing or null means
    "unmeasured", not zero: an empty mask measures exactly 0, and collapsing the two would
    hide the `no_defect` filter.
    """
    value = entry.get("defect_area_frac")
    return None if value is None else float(value)


def passes_eval_gate(defect_area_frac: float | None) -> bool:
    """The `gate` of every synthetic eval config: ``defect_area_frac != 0``.

    Missing and NaN pass, as they do in the pandas query the gate is written as: only a
    measured-empty mask is out. Defined on the value rather than on a row or a sample so
    the loader and every corpus writer decide it the same way — the cap below is
    only worth what this agreeing with `config.gate` is worth.
    """
    return defect_area_frac != 0


def sample_defects(
    samples: list[SynevadSample], max_defects: int, *, seed: int, category: str
) -> list[SynevadSample]:
    """At most ``max_defects`` gate-passing rows, drawn without replacement, in order.

    The draw is over the rows that survive the eval gate, so the cap pins what the metrics
    are actually computed on: capping the raw manifest instead would leave each arm a
    different number of defects again as soon as the gate dropped an uneven share of them.
    The rows it does not draw are gone, gate-failing ones included — that is the difference
    between this and the gate, which only hides them from a query.

    With ``max_defects`` or fewer gate-passing rows every row is returned, the gate-failing
    ones too: there is nothing to pin down, and dropping them here would change what an
    uncapped category is scored on for no reason. So a cap equalizes the arms only where
    every arm has at least that many — check the printed counts rather than assuming it.

    Seeded by ``seed`` and the category, never by ``config.seed``: the three sweep seeds
    must score the *same* eval set, or the spread across them stops being the model's and
    becomes the eval set's. Drawn from a local `default_rng` for the same reason
    `setup_and_evaluate` guards the global one — the training subset is drawn from that.
    """
    usable = [sample for sample in samples if passes_eval_gate(sample.defect_area_frac)]
    if len(usable) <= max_defects:
        return samples
    rng = np.random.default_rng([seed, zlib.crc32(category.encode())])
    keep = np.sort(rng.choice(len(usable), size=max_defects, replace=False))
    return [usable[i] for i in keep]


def stream_jsonl_file(path: str | Path):
    with Path(path).open(mode="rt", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def find_manifests(root: str | Path) -> list[Path]:
    """The generation manifests at ``root`` (a manifest file, or a directory of them)."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.glob("generation_manifest*"))


# The suffix the generator gives a crop of a frame
# (`synevad.synthesis.standalone_jobs.GenJob.roi_id`). Whatever cuts the detector's
# train/test crops of a cropped corpus has to spell it the same way.
_CROP_SUFFIX = re.compile(r"__crop\d+$")


def frame_id(stem: str) -> str:
    """The frame a file stem was cut from: the stem itself, minus any ``__crop<i>``."""
    return _CROP_SUFFIX.sub("", stem)


def leak_key(stem: str, group_pattern: str | None) -> str:
    """What two files must share to count as the same background.

    The frame by default. With ``group_pattern``, the regex's first group over the frame
    id — for corpora that photograph one physical object many times, where two different
    frames of it are still the same background.
    """
    frame = frame_id(stem)
    if group_pattern is None:
        return frame
    match = re.search(group_pattern, frame)
    if match is None:
        raise ValueError(
            f"group_pattern {group_pattern!r} does not match frame {frame!r}; the "
            "contamination check cannot group it"
        )
    return match.group(1)


def assert_no_train_contamination(
    root: str | Path, train_dir: str | Path, group_pattern: str | None = None
) -> None:
    """Fail if a frame used as a generation source is also in the detector's training set.

    Synthetic anomalies are edited onto clean train frames, so a source that stays in
    training lets the model memorize the exact background its defects sit on. The split is
    built by ``scripts/make_split.sh``; this only checks it still holds.

    Train files cut from a frame (``<frame>__crop<i>.png``) are compared by their frame.
    ``group_pattern`` widens "same frame" to "same group" — see :func:`leak_key`.
    """
    sources = {
        leak_key(str(entry["source_image_id"]), group_pattern)
        for manifest_path in find_manifests(root)
        for entry in stream_jsonl_file(manifest_path)
    }
    train = {leak_key(p.stem, group_pattern) for p in Path(train_dir).glob("*.png")}
    leaked = sorted(sources & train)
    if leaked:
        what = "generation source(s)" if group_pattern is None else "generation source group(s)"
        raise ValueError(
            f"{len(leaked)} {what} are still in the training set "
            f"{train_dir}: {leaked[:10]}{'...' if len(leaked) > 10 else ''} — rebuild the "
            "train/gen split (bash scripts/make_split.sh) and point MVTEC_PATH at it."
        )


# The manifest columns holding the composite and its pixel GT, per level. `full` is the
# whole source frame with the edit pasted back; `crop` is the edited window itself, at the
# generator's `resize_to` — what a corpus whose frames are far larger than the detector's
# input has to be scored on, or the defect is resampled away before PatchCore sees it.
LEVEL_PATHS: dict[str, tuple[str, str]] = {
    "full": ("full_blended_path", "mask_full_path"),
    "crop": ("blended_patch_path", "mask_path"),
}


def negative_samples(negatives_root: str | Path, level: str = "full") -> list[SynevadSample]:
    """Negative controls from a negatives run's manifests, all labelled defect-free.

    Every row must be a negative (``class: good`` and ``negative: true``): pointing this at a
    defect corpus by mistake would silently relabel defects as good, so it fails instead.
    The image is the composite at ``level``, like the defects it controls; the mask is None
    because a negative carries no defect pixels.
    """
    if level not in LEVEL_PATHS:
        raise ValueError(f"level must be one of {sorted(LEVEL_PATHS)}, got {level!r}")
    image_key, _ = LEVEL_PATHS[level]
    manifests = find_manifests(negatives_root)
    if not manifests:
        raise FileNotFoundError(
            f"No negative-control manifests were found at {negatives_root} — generate them "
            "with scripts/generate_standalone.py --negatives (or drop negatives_root)"
        )
    samples = []
    for manifest_path in manifests:
        for entry in stream_jsonl_file(manifest_path):
            if entry.get("class") not in OK_CLASS_NAMES or not entry.get("negative"):
                raise ValueError(
                    f"{manifest_path}: row {entry.get('image_id')!r} is not a negative control "
                    f"(class={entry.get('class')!r}); negatives_root must hold only a negatives run"
                )
            samples.append(
                SynevadSample(
                    image_path=Path(entry[image_key]),
                    mask_path=None,
                    label=0,
                    severity="none",
                    scorer_result="none",
                    class_name=NEGATIVE_CLASS_NAME,
                    mask_status=mask_status_value(entry),
                )
            )
    return samples


def preprocess_synevad_samples(
    root,
    filtering: ScorerResult,
    severity: Severity,
    good_dir: str | None,
    level: str = "full",
    negatives_root: str | None = None,
    max_defects: int | None = None,
    sample_seed: int = 0,
    category: str = "",
) -> list[SynevadSample]:
    if level not in LEVEL_PATHS:
        raise ValueError(f"level must be one of {sorted(LEVEL_PATHS)}, got {level!r}")
    image_key, mask_key = LEVEL_PATHS[level]

    samples = []
    # extract from all generation manifests at root
    manifests = find_manifests(root)
    if not manifests:
        raise FileNotFoundError(f"No manifests were found at {root}")

    for manifest_path in find_manifests(root):
        for entry in stream_jsonl_file(manifest_path):
            verdict = scorer_verdict(entry)
            if (
                filtering == ScorerResult.ALL or filtering == verdict
            ) and (severity == Severity.ALL or severity == entry["severity"]):
                sample = SynevadSample(
                    image_path=Path(entry[image_key]),
                    mask_path=Path(entry[mask_key]),
                    severity=entry["severity"],
                    scorer_result=verdict if verdict is not None else "none",
                    class_name=entry["class"],
                    label=int(entry["class"] not in OK_CLASS_NAMES),
                    mask_status=mask_status_value(entry),
                    reward_score=reward_margin(entry),
                    defect_area_frac=defect_area(entry),
                )
                samples.append(sample)

    # Before the defect-free frames are appended, so the cap counts defects and never
    # spends part of its budget on a `good` row. `scorer_result`/`severity` first: the cap
    # is over the rows the set is actually scored on, not over everything the manifest holds.
    if max_defects is not None:
        samples = sample_defects(
            samples, max_defects, seed=sample_seed, category=category
        )

    if negatives_root is not None:
        samples.extend(negative_samples(negatives_root, level))

    if good_dir is None:
        return samples

    for image_path in Path(good_dir).iterdir():
        sample = SynevadSample(
            image_path=image_path,
            mask_path=None,
            label=0,
            severity="none",
            scorer_result="none",
            mask_status="none",
            class_name="good",
        )
        samples.append(sample)

    return samples


def make_synevad_dataset(
    root: str,
    transform: T.Compose,
    scorer_result: ScorerResult = ScorerResult.ALL,
    severity: Severity = Severity.ALL,
    good_dir: str | None = None,
    train_dir: str | None = None,
    level: str = "full",
    group_pattern: str | None = None,
    negatives_root: str | None = None,
    max_defects: int | None = None,
    sample_seed: int = 0,
    category: str = "",
) -> ImageDataset:
    """The eval set one `data.synthetic` entry describes.

    ``max_defects`` pins how many synthetic anomalies the set carries, so two arms whose
    corpora are different sizes — FLUX at 120-200 gate-passing defects per category, DRAEM
    at exactly twice that — are compared on the same number of them. See `sample_defects`
    for what the draw does and does not equalize.
    """

    if train_dir is not None:
        assert_no_train_contamination(root, train_dir, group_pattern)
        if negatives_root is not None:
            # Negatives are inpainted onto generation sources too, so the same rule holds.
            assert_no_train_contamination(negatives_root, train_dir, group_pattern)

    samples = preprocess_synevad_samples(
        root,
        scorer_result,
        severity,
        good_dir,
        level,
        negatives_root,
        max_defects=max_defects,
        sample_seed=sample_seed,
        category=category,
    )

    dataset = ImageDataset(samples, transform)
    return dataset
