"""Run standalone generation: sequential two-phase or multi-GPU producer/consumer.

Klein prompt embeddings are encoded inside the first FLUX worker (default: first
gen GPU), written atomically to disk, then Qwen is unloaded before that worker
loads the DiT. Other gen workers wait for the ``.pt`` file and never load Qwen.
The mask process starts at the same time as the gen workers and polls
``crop_edited/``. ``skip_mask`` runs generation only so a sweep can blend the
previous category on the mask GPU while the next category generates.

With ``StandaloneRun.negatives`` set the same machinery produces **negative controls**
instead: each job's region is drawn and persisted first (:mod:`synevad.synthesis.regions`), Klein
inpaints only that region with a clean-surface prompt, and the blend pastes it back through
the region (:func:`synevad.synthesis.blend.blend_region`) with a defect-free manifest row.

The parent must not initialise CUDA before :func:`run_parallel` spawns.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from synevad.synthesis.blend import blend_candidate, blend_region
from synevad.synthesis.crops import crop_image
from synevad.synthesis.devices import GpuAssignment, cap_cpu_threads, hide_parent_cuda, pin_visible_gpu
from synevad.synthesis.generate import (
    edit_image,
    encode_prompt_texts,
    inpaint_image,
    load_pipeline,
    load_prompt_embeds,
    load_text_encoder,
    save_prompt_embeds,
    uses_klein,
)
from synevad.synthesis.manifest import append_manifest, read_manifest, write_manifest
from synevad.synthesis.persist import (
    ArtifactDirs,
    persist_composite,
    raw_edit_path,
    region_path,
    write_raw_edit,
    write_region,
)
from synevad.synthesis.regions import inpaint_support, region_spec, resolve_shape, sample_region
from synevad.synthesis.standalone_jobs import GenJob, candidate_id, unique_prompts


@dataclass
class StandaloneRun:
    """Picklable run settings shared by generation and mask workers."""

    run_id: str
    dirs: ArtifactDirs
    manifest_path: Path
    resize_to: int | tuple[int, int]  # square side, or (w, h) for a non-square frame
    steps: int
    guidance: float
    base_model: str
    quantization: str
    cpu_offload: bool
    blending: dict
    masks: dict
    prompt_embeds_path: Path | None = None
    # The config's `negatives:` block when this run generates negative controls; None for defects.
    negatives: dict | None = None

    @property
    def task(self) -> str:
        return "inpaint" if self.negatives is not None else "edit"


def apply_mask_device(masks: dict, device: str) -> dict:
    """Copy ``device`` onto the encoder sub-blocks (and ``masks.device``)."""
    out = dict(masks)
    out["device"] = device
    for key in ("resnet", "dinov3", "post_blend"):
        block = out.get(key)
        if block:
            block = dict(block)
            block["device"] = device
            out[key] = block
    return out


def _frames(
    job: GenJob, resize_to: int | tuple[int, int], cache: dict
) -> tuple[Image.Image, Image.Image]:
    """``(source, clean_crop)``, cached per source path *and* crop box.

    The source frame decodes once; the clean crop is per box, so a multi-crop run of one
    frame does not re-read the file for each window.
    """
    key = (str(job.source_path), job.box)
    hit = cache.get(key)
    if hit is not None:
        return hit
    src_key = str(job.source_path)
    source = cache.get(src_key)
    if source is None:
        source = Image.open(job.source_path).convert("RGB")
        cache[src_key] = source
    clean = crop_image(source, job.box, resize_to)
    cache[key] = (source, clean)
    return source, clean


def _load_mask_models(masks: dict) -> tuple:
    """``(semantic_fn, post_semantic_fn)`` — the paste seed and the post-blend GT map."""
    from synevad.synthesis.semantic import load_change_fns

    return load_change_fns(masks)


def _blend_one(
    job: GenJob,
    run: StandaloneRun,
    *,
    cache: dict,
    semantic_fn,
    post_semantic_fn=None,
) -> dict[str, Any]:
    """Load the raw edit, composite, persist, return one manifest row."""
    from synevad.synthesis.profile import span

    name = candidate_id(run.run_id, job)
    with span("blend_one", image_id=name):
        with span("blend.frames"):
            source, clean = _frames(job, run.resize_to, cache)
            raw_path = raw_edit_path(run.dirs, name)
            edit_crop = Image.open(raw_path).convert("RGB")
        region_stats: dict[str, Any] = {}
        with span("blend.composite"):
            if run.negatives is not None:
                full, patch, mask, region_stats = blend_region(
                    source,
                    clean,
                    edit_crop,
                    job.box,
                    negative_region(job, run, clean),
                    blending=run.blending,
                    masks=run.masks,
                    seed=job.seed,
                    resize_to=run.resize_to,
                )
            else:
                full, patch, mask = blend_candidate(
                    source,
                    clean,
                    edit_crop,
                    job.box,
                    blending=run.blending,
                    masks=run.masks,
                    seed=job.seed,
                    resize_to=run.resize_to,
                    stage=job.prompt.stage,
                    category=job.prompt.category,
                    semantic_fn=semantic_fn,
                    post_semantic_fn=post_semantic_fn,
                )
        with span("blend.persist"):
            paths = persist_composite(
                run.dirs,
                name=name,
                roi_id=job.roi_id,
                source_path=job.source_path,
                box=job.box,
                source_size=source.size,
                clean_crop=clean,
                full=full,
                patch=patch,
                mask=mask,
                raw_path=raw_path,
            )
        with span("blend.area"):
            from synevad.metrics.area import area_fields

            area_columns = area_fields(mask.binary)
    row = {
        "image_id": name,
        "source_image_id": job.source_image_id,
        "roi_id": job.roi_id,
        "generation_id": run.run_id,
        "category": job.prompt.category,
        "class": job.prompt.mode,
        "severity": job.prompt.stage,
        "prompt_id": f"{job.prompt.mode}_{job.prompt.stage}",
        "prompt": job.prompt.prompt,
        "seed": job.seed,
        "guidance_scale": float(run.guidance),
        "num_diffusion_steps": int(run.steps),
        "attempt_index": job.attempt_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scorer": None,
        "reward_score": None,
        "reward_edit_score": None,
        "reward_noop_score": None,
        **mask.manifest_fields(),
        **area_columns,
        "crop_box": list(job.box),
        **paths,
    }
    if run.negatives is not None:
        row.update(negative_row_fields(job, run, region_stats))
    return row


def negative_row_fields(job: GenJob, run: StandaloneRun, region_stats: dict) -> dict[str, Any]:
    """The manifest columns that make a row a negative control.

    ``class: good`` is what the eval labels defect-free (``synevad.data.synevad.OK_CLASS_NAMES``);
    ``severity: none`` matches the real defect-free frames. The defect cell the negative mirrors
    stays recoverable as ``region_mode`` / ``region_stage`` — the resume and quota keys read
    those (:func:`synevad.synthesis.standalone_jobs.row_key`, :func:`synevad.synthesis.sweep.generated_counts`).
    """
    name = candidate_id(run.run_id, job)
    block = run.negatives or {}
    return {
        "negative": True,
        "class": "good",
        "severity": "none",
        "prompt_id": f"negative_{job.prompt.mode}_{job.prompt.stage}",
        "region_mode": job.prompt.mode,
        "region_stage": job.prompt.stage,
        "region_shape": resolve_shape(job.prompt.region_shape, job.prompt.mode, block),
        "region_path": str(region_path(run.dirs, name).resolve()),
        "inpaint_strength": float(block.get("strength", 1.0)),
        **region_stats,
    }


def prompt_embeds_file(run: StandaloneRun) -> Path:
    return run.manifest_path.parent / f"{run.run_id}_prompt_embeds.pt"


def drop_stale_prompt_embeds(path: Path, prompts: list[str]) -> bool:
    """Delete a cached ``.pt`` that lacks any of ``prompts``, so the encoding worker rewrites it.

    The file is per run id, not per prompt set: a second invocation under one run id with other
    ``--modes``/``--stages`` would reuse the first one's embeds and fail on its first job
    (``missing prompt embeds``). Called before any worker starts, so sibling FLUX workers wait
    for the new file instead of loading the old one. True if it deleted the file.
    """
    if not path.is_file():
        return False
    missing = [p for p in prompts if p not in load_prompt_embeds(path)]
    if not missing:
        return False
    print(
        f"[embed] {path.name} lacks {len(missing)} of {len(prompts)} prompt(s); re-encoding",
        flush=True,
    )
    path.unlink()
    return True


def wait_for_prompt_embeds(path: Path, *, timeout_s: float = 1800.0) -> None:
    """Block until another gen worker has atomically written ``path``."""
    import time

    from synevad.synthesis.profile import span

    deadline = time.monotonic() + timeout_s
    with span("wait_for_embeds", kind="block"):
        while time.monotonic() < deadline:
            if path.is_file() and path.stat().st_size > 0:
                return
            time.sleep(0.25)
    raise SystemExit(f"timed out waiting for prompt embeds {path}")


def shard_jobs(jobs: list[GenJob], index: int, n: int) -> list[GenJob]:
    """Round-robin slice ``jobs`` for blend worker ``index`` of ``n``."""
    if n <= 1:
        return list(jobs)
    return [job for i, job in enumerate(jobs) if i % n == index]


def mask_proc_name(gpu: int, index: int, n: int) -> str:
    if n == 1:
        return f"mask-gpu{gpu}"
    return f"mask-gpu{gpu}-w{index}"


def embedder_gpu(assignment: GpuAssignment) -> int:
    """Which FLUX worker encodes Qwen prompts (never the mask GPU in parallel mode)."""
    if assignment.embed_gpu in assignment.gen_gpus:
        return assignment.embed_gpu
    return assignment.gen_gpus[0]


def _prompt_list(jobs: list[GenJob]) -> list[str]:
    prompts = unique_prompts(jobs)
    if "" not in prompts:
        prompts = list(prompts) + [""]
    return prompts


def _write_prompt_embeds(run: StandaloneRun, prompts: list[str], out_path: Path) -> None:
    """Encode unique prompts; caller must already pin the embed GPU.

    The tokenizer only exists here — the FLUX workers load with ``text_encoder=None`` and
    wait on the atomically written ``.pt`` (:func:`wait_for_prompt_embeds`).
    """
    print(f"[embed] Qwen3: {len(prompts)} unique prompt(s) -> {out_path}", flush=True)
    tokenizer, text_encoder = load_text_encoder(
        run.base_model, quantization=run.quantization
    )
    mapping = encode_prompt_texts(tokenizer, text_encoder, prompts)
    save_prompt_embeds(out_path, mapping)
    del text_encoder
    free_gpu()


def embed_worker_body(gpu: int, run: StandaloneRun, prompts: list[str], out_path: Path) -> None:
    from synevad.synthesis.profile import configure, span

    configure("embed", gpu)
    print(f"[embed] worker on cuda:{gpu}", flush=True)
    with span("embed_prompts", n=len(prompts)):
        _write_prompt_embeds(run, prompts, out_path)


def negative_region(job: GenJob, run: StandaloneRun, clean: Image.Image) -> Image.Image:
    """The job's region: read back if already persisted, else drawn from the job seed and written.

    Shape comes from the prompt entry (or ``negatives.shapes``), size from the job's severity
    stage, placement from the clean crop — :mod:`synevad.synthesis.regions`.
    """
    name = candidate_id(run.run_id, job)
    path = region_path(run.dirs, name)
    if path.is_file():
        return Image.open(path).convert("L")
    block = run.negatives or {}
    shape = resolve_shape(job.prompt.region_shape, job.prompt.mode, block)
    spec = region_spec(block, shape=shape, stage=job.prompt.stage, mode=job.prompt.mode)
    region = sample_region(clean, spec, seed=job.seed)
    image = region.image()
    write_region(run.dirs, name, image)
    return image


def _generate_one(
    job: GenJob,
    run: StandaloneRun,
    pipe,
    cache: dict,
    embed_cache: dict | None = None,
) -> Path:
    """Edit (or, for negatives, inpaint) with FLUX and write the raw crop; returns the path."""
    from synevad.synthesis.profile import span

    name = candidate_id(run.run_id, job)
    with span("generate_one", image_id=name, mode=job.prompt.mode, stage=job.prompt.stage):
        _, clean = _frames(job, run.resize_to, cache)
        embeds = None
        if embed_cache is not None:
            embeds = embed_cache.get(job.prompt.prompt)
            if embeds is None:
                raise KeyError(
                    f"missing prompt embeds for {job.prompt.mode}/{job.prompt.stage}"
                )
        if run.negatives is not None:
            import numpy as np

            region = negative_region(job, run, clean)
            support = inpaint_support(
                np.asarray(region) > 127,
                int(run.masks["dilate_px"]),
                int(run.masks["alpha_feather_px"]),
            )
            edit_crop = inpaint_image(
                pipe,
                clean,
                Image.fromarray(support.astype(np.uint8) * 255, mode="L"),
                job.prompt.prompt,
                steps=int(run.steps),
                guidance=float(run.guidance),
                seed=job.seed,
                resize_to=run.resize_to,
                strength=float((run.negatives or {}).get("strength", 1.0)),
                prompt_embeds=embeds,
            )
        else:
            edit_crop = edit_image(
                pipe,
                clean,
                job.prompt.prompt,
                steps=int(run.steps),
                guidance=float(run.guidance),
                seed=job.seed,
                resize_to=run.resize_to,
                prompt_embeds=embeds,
            )
        return write_raw_edit(run.dirs, name, edit_crop)


def free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential(
    *,
    need_generate: list[GenJob],
    blend_only: list[GenJob],
    assignment: GpuAssignment,
    run: StandaloneRun,
    manifest: list[dict],
    skip_mask: bool = False,
) -> list[dict]:
    """Generate every missing raw edit, then blend everything, on one GPU."""
    from synevad.synthesis.profile import configure, span

    physical = assignment.gen_gpus[0]
    configure("gen", physical)
    device = pin_visible_gpu(physical)
    cache: dict = {}
    embed_cache: dict | None = None
    skip_text_encoder = False
    if need_generate:
        print(
            f"[gen] sequential on cuda:{physical}: {len(need_generate)} to generate, "
            f"{len(blend_only)} blend-only",
            flush=True,
        )
        if uses_klein(run.base_model):
            out_path = prompt_embeds_file(run)
            _write_prompt_embeds(run, _prompt_list(need_generate), out_path)
            run.prompt_embeds_path = out_path
            embed_cache = load_prompt_embeds(out_path)
            skip_text_encoder = True
        with span("load_pipeline"):
            pipe = load_pipeline(
                run.base_model,
                device=device,
                quantization=run.quantization,
                cpu_offload=run.cpu_offload,
                include_text_encoder=not skip_text_encoder,
                task=run.task,
            )
        for i, job in enumerate(need_generate):
            name = candidate_id(run.run_id, job)
            print(
                f"[gen] {i + 1}/{len(need_generate)} {name} "
                f"({job.prompt.mode}/{job.prompt.stage})",
                flush=True,
            )
            _generate_one(job, run, pipe, cache, embed_cache)
        del pipe
        free_gpu()

    if skip_mask:
        return manifest

    run.masks = apply_mask_device(run.masks, device)
    semantic_fn, post_semantic_fn = (
        (None, None) if run.negatives is not None else _load_mask_models(run.masks)
    )
    to_blend = list(need_generate) + list(blend_only)
    for i, job in enumerate(to_blend):
        name = candidate_id(run.run_id, job)
        print(
            f"[blend] {i + 1}/{len(to_blend)} {name} "
            f"({job.prompt.mode}/{job.prompt.stage})",
            flush=True,
        )
        row = _blend_one(
            job,
            run,
            cache=cache,
            semantic_fn=semantic_fn,
            post_semantic_fn=post_semantic_fn,
        )
        manifest.append(row)
        write_manifest(run.manifest_path, manifest)
    return manifest


def gen_worker_body(
    gpu: int,
    gen_q: mp.Queue,
    run: StandaloneRun,
    encode_prompts: list[str] | None = None,
) -> None:
    import time

    from synevad.synthesis.profile import configure, span

    configure("gen", gpu)
    print(f"[gen] worker on cuda:{gpu}", flush=True)
    if encode_prompts:
        out_path = run.prompt_embeds_path or prompt_embeds_file(run)
        if out_path.is_file():
            print(f"[embed] reuse {out_path}", flush=True)
        else:
            with span("embed_prompts", n=len(encode_prompts)):
                _write_prompt_embeds(run, encode_prompts, out_path)
        run.prompt_embeds_path = out_path
    elif run.prompt_embeds_path is not None:
        wait_for_prompt_embeds(run.prompt_embeds_path)
    pipe = None
    embed_cache: dict | None = None
    cache: dict = {}
    while True:
        with span("wait_for_job", kind="block"):
            job = gen_q.get()
        if job is None:
            break
        if pipe is None:
            skip_text_encoder = run.prompt_embeds_path is not None
            with span("load_pipeline"):
                pipe = load_pipeline(
                    run.base_model,
                    device="cuda",
                    quantization=run.quantization,
                    cpu_offload=run.cpu_offload,
                    include_text_encoder=not skip_text_encoder,
                    task=run.task,
                )
                embed_cache = (
                    load_prompt_embeds(run.prompt_embeds_path)
                    if run.prompt_embeds_path
                    else None
                )
        name = candidate_id(run.run_id, job)
        print(
            f"[gen] cuda:{gpu} start {name} ({job.prompt.mode}/{job.prompt.stage})",
            flush=True,
        )
        t0 = time.perf_counter()
        _generate_one(job, run, pipe, cache, embed_cache)
        print(
            f"[gen] cuda:{gpu} done {name} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    if pipe is not None:
        del pipe
        free_gpu()


def mask_worker_body(
    gpu: int,
    jobs: list[GenJob],
    gens_done: Any,
    run: StandaloneRun,
    initial_manifest: list[dict],
    manifest_lock: Any = None,
) -> None:
    """Blend each job as soon as its raw edit exists; never block a gen worker."""
    import time

    from synevad.synthesis.profile import configure, emit, span

    configure("mask", gpu)
    print(f"[blend] worker on cuda:{gpu} ({len(jobs)} job(s))", flush=True)
    run.masks = apply_mask_device(run.masks, "cuda")
    with span("load_mask_models"):
        semantic_fn, post_semantic_fn = (
            (None, None) if run.negatives is not None else _load_mask_models(run.masks)
        )
    cache: dict = {}
    manifest = list(initial_manifest)
    pending = {candidate_id(run.run_id, job): job for job in jobs}
    idle_t0: float | None = None
    while pending:
        ready = [
            name
            for name, job in pending.items()
            if raw_edit_path(run.dirs, name).is_file()
        ]
        if ready and idle_t0 is not None:
            emit("wait_for_raw", dur_ms=(time.perf_counter() - idle_t0) * 1000.0, kind="block")
            idle_t0 = None
        for name in ready:
            job = pending.pop(name)
            print(
                f"[blend] cuda:{gpu} {name} ({job.prompt.mode}/{job.prompt.stage})",
                flush=True,
            )
            row = _blend_one(
                job,
                run,
                cache=cache,
                semantic_fn=semantic_fn,
                post_semantic_fn=post_semantic_fn,
            )
            if manifest_lock is None:
                manifest.append(row)
                write_manifest(run.manifest_path, manifest)
            else:
                with manifest_lock:
                    append_manifest(run.manifest_path, [row])
        if not pending:
            break
        if gens_done.is_set():
            still_ready = any(
                raw_edit_path(run.dirs, name).is_file() for name in pending
            )
            if still_ready:
                continue
            missing = ", ".join(sorted(pending))
            raise SystemExit(f"[blend] raw edits never appeared: {missing}")
        if idle_t0 is None:
            idle_t0 = time.perf_counter()
        gens_done.wait(timeout=0.5)


def run_parallel(
    *,
    need_generate: list[GenJob],
    blend_only: list[GenJob],
    assignment: GpuAssignment,
    run: StandaloneRun,
    manifest: list[dict],
    skip_mask: bool = False,
) -> list[dict]:
    """FLUX on ``assignment.gen_gpus``; mask/blend on ``assignment.mask_gpus``.

    Klein prompt embeds are encoded inside the first FLUX worker (Qwen then DiT on
    the same card). Sibling workers wait for the ``.pt`` file. Mask workers start
    together with gen unless ``skip_mask`` (sweep overlaps the next category's
    generation). Jobs are round-robin sharded across blend workers. When
    ``need_generate`` is empty (blend-only resume), gen workers are not started.
    """
    from synevad.synthesis.cuda_worker import gen_worker, mask_worker
    from synevad.synthesis.profile import configure, span, start_smi_sampler, stop_smi_sampler

    configure("parent")
    cap_cpu_threads(1)
    start_smi_sampler()
    try:
        with hide_parent_cuda():
            encode_on = embedder_gpu(assignment) if assignment.gen_gpus else None
            klein = bool(need_generate) and uses_klein(run.base_model)
            if klein:
                run.prompt_embeds_path = prompt_embeds_file(run)
                if assignment.embed_gpu not in assignment.gen_gpus:
                    print(
                        f"[embed] embed GPU {assignment.embed_gpu} is not a gen worker; "
                        f"encoding on cuda:{encode_on} so the mask card stays free",
                        flush=True,
                    )

            ctx = mp.get_context("spawn")
            gens_done = ctx.Event()
            to_blend = list(need_generate) + list(blend_only)
            encode_prompts = _prompt_list(need_generate) if klein else None
            if encode_prompts:
                drop_stale_prompt_embeds(run.prompt_embeds_path, encode_prompts)

            gen_procs: list = []
            if need_generate:
                gen_q: mp.Queue = ctx.Queue()
                for job in need_generate:
                    gen_q.put(job)
                for _ in assignment.gen_gpus:
                    gen_q.put(None)
                gen_procs = [
                    ctx.Process(
                        target=gen_worker,
                        args=(
                            gpu,
                            gen_q,
                            run,
                            encode_prompts if gpu == encode_on else None,
                        ),
                        name=f"flux-gpu{gpu}",
                    )
                    for gpu in assignment.gen_gpus
                ]

            mask_procs: list = []
            if not skip_mask:
                n_mask = len(assignment.mask_gpus)
                lock = ctx.Lock() if n_mask > 1 else None
                for i, gpu in enumerate(assignment.mask_gpus):
                    shard = shard_jobs(to_blend, i, n_mask)
                    if not shard and n_mask > 1:
                        continue
                    mask_procs.append(
                        ctx.Process(
                            target=mask_worker,
                            args=(gpu, shard, gens_done, run, manifest, lock),
                            name=mask_proc_name(gpu, i, n_mask),
                        )
                    )
            if gen_procs and mask_procs:
                print(
                    f"[gpus] starting {len(gen_procs)} gen workers {assignment.gen_gpus} "
                    f"+ {len(mask_procs)} mask {assignment.mask_gpus}",
                    flush=True,
                )
            elif gen_procs:
                print(
                    f"[gpus] starting {len(gen_procs)} gen workers {assignment.gen_gpus} "
                    f"(mask deferred)",
                    flush=True,
                )
            else:
                print(
                    f"[gpus] blend-only: skip gen workers, mask {assignment.mask_gpus}",
                    flush=True,
                )
            for proc in gen_procs:
                proc.start()
            for proc in mask_procs:
                proc.start()
            if not gen_procs:
                gens_done.set()

            failed = []
            with span("gen_join"):
                for proc in gen_procs:
                    proc.join()
                    if proc.exitcode not in (0, None):
                        failed.append(f"{proc.name} exit {proc.exitcode}")
            gens_done.set()
            if mask_procs:
                with span("mask_join"):
                    for proc in mask_procs:
                        proc.join()
                        if proc.exitcode not in (0, None):
                            failed.append(f"{proc.name} exit {proc.exitcode}")
            if failed:
                raise SystemExit("generation workers failed: " + "; ".join(failed))

            return read_manifest(run.manifest_path)
    finally:
        stop_smi_sampler()


def run_jobs(
    *,
    need_generate: list[GenJob],
    blend_only: list[GenJob],
    assignment: GpuAssignment,
    run: StandaloneRun,
    manifest: list[dict],
    skip_mask: bool = False,
) -> list[dict]:
    """Dispatch sequential or parallel depending on ``assignment``."""
    if not need_generate and not blend_only:
        return manifest
    if skip_mask and not need_generate:
        return manifest
    if assignment.sequential:
        return run_sequential(
            need_generate=need_generate,
            blend_only=blend_only,
            assignment=assignment,
            run=run,
            manifest=manifest,
            skip_mask=skip_mask,
        )
    return run_parallel(
        need_generate=need_generate,
        blend_only=blend_only,
        assignment=assignment,
        run=run,
        manifest=manifest,
        skip_mask=skip_mask,
    )
