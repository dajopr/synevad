"""Standalone generation jobs: slim specs, expansion, resume splits.

:class:`GenJob` holds paths and ids only (no PIL images) so it pickles cheaply onto
worker queues. :func:`build_jobs` enumerates source × crop × prompt × attempt in
category-then-source order, the crops coming from a :class:`synevad.synthesis.crops.CropPlan`
(one whole-frame box unless the config asks for real crops).
:func:`partition_pending` splits remaining jobs into those that still need FLUX and
those that only need blending (raw edit already on disk).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from synevad.synthesis.crops import Box, CropPlan
from synevad.synthesis.persist import ArtifactDirs, raw_edit_path
from synevad.synthesis.standalone_prompts import StandalonePrompt
from synevad.synthesis.sweep import STAGE_ORDER

_STAGE_RANK = {name: i for i, name in enumerate(STAGE_ORDER)}


@dataclass(frozen=True)
class GenJob:
    """One independent (source × crop × prompt × attempt) edit from a clean frame."""

    source_path: Path
    source_image_id: str
    box: Box
    prompt: StandalonePrompt
    attempt_index: int
    seed: int
    crop_index: int = 0

    @property
    def roi_id(self) -> str:
        return f"{self.source_image_id}__crop{self.crop_index}"

    @property
    def stem(self) -> str:
        return (
            f"{self.roi_id}__{self.prompt.mode}__{self.prompt.stage}"
            f"__a{self.attempt_index}"
        )

    @property
    def sort_key(self) -> tuple:
        stage_rank = _STAGE_RANK.get(self.prompt.stage, len(_STAGE_RANK))
        return (
            self.prompt.category,
            self.source_image_id,
            self.crop_index,
            self.prompt.mode,
            stage_rank,
            self.prompt.stage,
            self.attempt_index,
        )


def candidate_id(run_id: str, job: GenJob) -> str:
    return f"{run_id}__{job.stem}"


def unique_prompts(jobs: list[GenJob]) -> list[str]:
    """Distinct prompt strings, first-seen order (shared across sources / attempts)."""
    seen: set[str] = set()
    out: list[str] = []
    for job in jobs:
        text = job.prompt.prompt
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def job_key(job: GenJob) -> tuple[str, str, str, int]:
    """Logical identity of a generation job, independent of run_id / filenames.

    Keyed on ``roi_id``, not ``source_image_id``: with more than one crop per frame the
    frame alone no longer identifies a job, and ``--continue`` would drop every crop
    after the first as already done.
    """
    return (
        job.roi_id,
        job.prompt.mode,
        job.prompt.stage,
        job.attempt_index,
    )


def row_key(row: dict) -> tuple[str, str, str, int]:
    """Logical identity of a persisted manifest row (matches :func:`job_key`).

    ``roi_id`` has been written on every row since the first standalone run; the
    whole-frame fallback is only for a hand-edited manifest.

    A negative control is stored as ``class: good`` / ``severity: none`` (so the eval labels it
    defect-free), and records the defect cell it stands in for as ``region_mode`` /
    ``region_stage``. Those are its identity: keyed on class and severity, every negative of an
    ROI would collide and ``--continue`` would regenerate all but one of them.
    """
    roi_id = row.get("roi_id") or f"{row['source_image_id']}__crop0"
    return (
        str(roi_id),
        str(row.get("region_mode") or row["class"]),
        str(row.get("region_stage") or row["severity"]),
        int(row.get("attempt_index", 0)),
    )


def _prompt_index(prompts: list[StandalonePrompt]) -> dict[tuple[str, str, str], int]:
    """Stable index per (category, mode, stage) from a category-then-mode sort."""
    ordered = sorted(
        prompts,
        key=lambda p: (
            p.category,
            p.mode,
            _STAGE_RANK.get(p.stage, len(_STAGE_RANK)),
            p.stage,
        ),
    )
    return {(p.category, p.mode, p.stage): i for i, p in enumerate(ordered)}


def build_jobs(
    sources: list[Path],
    prompts: list[StandalonePrompt],
    *,
    base_seed: int,
    num_per_source: int,
    crops: CropPlan | None = None,
) -> list[GenJob]:
    """Expand sources × crops × prompts × num_per_source, sorted category then source.

    Each attempt uses ``base_seed + crop_index * 1_000_000 + prompt_index * 1000 +
    attempt``. Prompt indices are assigned from a category/mode/stage sort so seeds do
    not depend on filesystem order of the prompt files, and the crop stride is far above
    the prompt stride so two crops of one frame never share a seed. Crop 0 keeps the
    seeds a whole-frame run has always used. The source is opened only for its size;
    pixels are not kept on the job.
    """
    n = max(1, int(num_per_source))
    plan = crops or CropPlan()
    pindex = _prompt_index(prompts)
    jobs: list[GenJob] = []
    for src in sources:
        with Image.open(src) as im:
            size = (im.width, im.height)
        sid = src.stem
        for ci, box in enumerate(plan.boxes(size)):
            for prompt in prompts:
                pi = pindex[(prompt.category, prompt.mode, prompt.stage)]
                for attempt in range(n):
                    jobs.append(
                        GenJob(
                            source_path=src,
                            source_image_id=sid,
                            box=box,
                            prompt=prompt,
                            attempt_index=attempt,
                            seed=int(base_seed) + ci * 1_000_000 + pi * 1000 + attempt,
                            crop_index=ci,
                        )
                    )
    jobs.sort(key=lambda j: j.sort_key)
    return jobs


def filter_completed(
    jobs: list[GenJob], existing_rows: list[dict], run_id: str
) -> tuple[list[GenJob], list[dict]]:
    """Skip jobs whose image_id is already on disk in this run's manifest."""
    existing_ids = {r["image_id"] for r in existing_rows}
    pending: list[GenJob] = []
    redo_ids: set[str] = set()
    for job in jobs:
        cid = candidate_id(run_id, job)
        if cid in existing_ids:
            continue
        pending.append(job)
        redo_ids.add(cid)
    kept = [r for r in existing_rows if r["image_id"] not in redo_ids]
    return pending, kept


def filter_completed_continue(
    jobs: list[GenJob], all_rows: list[dict]
) -> tuple[list[GenJob], int]:
    """Skip jobs whose logical key already exists in any out_dir manifest."""
    done = {row_key(r) for r in all_rows if "source_image_id" in r and "class" in r}
    pending = [job for job in jobs if job_key(job) not in done]
    return pending, len(jobs) - len(pending)


def partition_pending(
    jobs: list[GenJob],
    *,
    run_id: str,
    dirs: ArtifactDirs,
) -> tuple[list[GenJob], list[GenJob]]:
    """Split unfinished jobs into ``(need_generate, blend_only)``.

    A job with a ``crop_edited/{image_id}.png`` already on disk skips FLUX and is only
    blended. Manifest presence is the caller's job — pass only the still-pending jobs.
    """
    need_generate: list[GenJob] = []
    blend_only: list[GenJob] = []
    for job in jobs:
        name = candidate_id(run_id, job)
        if raw_edit_path(dirs, name).is_file():
            blend_only.append(job)
        else:
            need_generate.append(job)
    return need_generate, blend_only
