"""Spawn entrypoints: pin GPU and CPU-thread caps before numpy/torch load.

``multiprocessing`` spawn imports this module first, then unpickles arguments
(which pulls in ``standalone_run`` / numpy). Thread env vars must already be in
the inherited environment — :func:`synevad.synthesis.devices.cap_cpu_threads` is called in the
parent before ``Process.start``. ``CUDA_VISIBLE_DEVICES`` is set here, still before
any ``import torch``.
"""

from __future__ import annotations

from pathlib import Path


def _bootstrap(gpu: int) -> None:
    from synevad.synthesis.devices import pin_visible_gpu

    pin_visible_gpu(gpu)


def gen_worker(gpu: int, gen_q, run, encode_prompts=None) -> None:
    from synevad.synthesis.standalone_run import gen_worker_body

    _bootstrap(gpu)
    gen_worker_body(gpu, gen_q, run, encode_prompts=encode_prompts)


def mask_worker(
    gpu: int, jobs: list, gens_done, run, initial_manifest: list, lock=None
) -> None:
    from synevad.synthesis.standalone_run import mask_worker_body

    _bootstrap(gpu)
    mask_worker_body(gpu, jobs, gens_done, run, initial_manifest, manifest_lock=lock)


def embed_worker(gpu: int, run, prompts: list[str], out_path: Path) -> None:
    from synevad.synthesis.standalone_run import embed_worker_body

    _bootstrap(gpu)
    embed_worker_body(gpu, run, prompts, out_path)
