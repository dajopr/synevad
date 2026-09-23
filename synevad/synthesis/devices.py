"""Parse and split GPU ids for standalone generation.

Generation workers occupy every listed GPU except those reserved for mask/blend
workers. Several blend workers are supported: ``--mask-gpu 2,3`` pins one
process per card, and ``--mask-workers 2`` can run two processes on the same card.
Klein prompt embeddings (Qwen3) run on the first FLUX worker before it loads the DiT,
so the mask GPU(s) can blend while other workers generate. A single GPU is sequential:
embed, generate, then blend, on the same device — unless ``--mask-workers`` asks for more
than one blend process, which then share the card with the FLUX worker. Pure stdlib — no torch — so assignment
is CPU-testable before any CUDA init.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class GpuAssignment:
    """Where FLUX workers, Qwen embeddings, and the mask/blend worker(s) should run."""

    gen_gpus: list[int]
    mask_gpu: int
    sequential: bool
    embed_gpu: int
    mask_gpus: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mask_gpus:
            object.__setattr__(self, "mask_gpus", [self.mask_gpu])

    @property
    def mask_device(self) -> str:
        return f"cuda:{self.mask_gpu}"

    @property
    def embed_device(self) -> str:
        return f"cuda:{self.embed_gpu}"

    def gen_device(self, gpu: int) -> str:
        return f"cuda:{gpu}"


def parse_gpu_ids(raw: str | None) -> list[int] | None:
    """Parse ``0,1,2,3`` into distinct non-negative ints. ``None`` / empty → ``None``."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid GPU id {part!r} in {raw!r}") from exc
        if gpu < 0:
            raise ValueError(f"GPU id must be >= 0, got {gpu}")
        ids.append(gpu)
    if not ids:
        return None
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate GPU ids: {ids}")
    return ids


def parse_device_id(raw: str | int | None) -> int | None:
    """Accept ``3``, ``cuda:3``, or ``None``."""
    if raw is None:
        return None
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"GPU id must be >= 0, got {raw}")
        return raw
    text = str(raw).strip()
    if not text:
        return None
    if text.lower().startswith("cuda:"):
        text = text.split(":", 1)[1]
    try:
        gpu = int(text)
    except ValueError as exc:
        raise ValueError(f"invalid device {raw!r}") from exc
    if gpu < 0:
        raise ValueError(f"GPU id must be >= 0, got {gpu}")
    return gpu


def parse_gpu_id_list(raw) -> list[int] | None:
    """Accept a YAML list of ints, a single int, ``cuda:N``, or a comma-separated string."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"generation.devices must be a list or comma-separated string, got {raw!r}"
        )
    if isinstance(raw, int):
        return parse_gpu_ids(str(raw))
    if isinstance(raw, str):
        return parse_gpu_ids(raw)
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(
            f"generation.devices must be a list or comma-separated string, got {raw!r}"
        ) from exc
    return parse_gpu_ids(",".join(str(parse_device_id(x)) for x in items))


def assign_devices(
    gpu_ids: list[int],
    mask_gpu: int | str | list | None = None,
    embed_gpu: int | None = None,
    mask_workers: int | None = None,
) -> GpuAssignment:
    """Split ``gpu_ids`` into FLUX workers and reserved mask GPU(s).

    * One GPU, or ``mask_gpu`` equal to the only id → sequential on that device, unless
      ``mask_workers`` > 1: then one FLUX worker and that many blend processes share it.
    * Several GPUs, ``mask_gpu`` omitted → last id is reserved for masking.
    * ``mask_gpu`` may be several ids (``2,3`` / ``[2, 3]``) and may sit outside
      ``gpu_ids`` (config lists only generation devices).
    * ``mask_workers`` defaults to ``len(mask_gpus)``. A larger value cycles the
      reserved cards (``--mask-workers 2`` with one mask GPU → two blend processes).
    * ``embed_gpu`` defaults to the first generation GPU. Qwen encoding runs inside
      that FLUX worker (then unloads) so the mask card can blend in parallel.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must list at least one device")
    if mask_gpu is None:
        reserved = [gpu_ids[-1]]
    else:
        parsed = parse_gpu_id_list(mask_gpu)
        if not parsed:
            raise ValueError("mask_gpu must list at least one device")
        reserved = parsed
    reserved_set = set(reserved)
    gen = [g for g in gpu_ids if g not in reserved_set]
    if not gen:
        gen = [reserved[0]]
        # Blending is CPU-bound (~11 s a row against ~3 s a FLUX edit on one card), so
        # asking for several blend workers shares the card with the FLUX worker.
        sequential = mask_workers is None or int(mask_workers) <= 1
    else:
        sequential = False
    n_workers = 1 if sequential else (
        int(mask_workers) if mask_workers is not None else len(reserved)
    )
    if n_workers < 1:
        raise ValueError(f"mask_workers must be >= 1, got {n_workers}")
    worker_gpus = [reserved[i % len(reserved)] for i in range(n_workers)]
    embed = gen[0] if embed_gpu is None else int(embed_gpu)
    return GpuAssignment(
        gen_gpus=list(gen),
        mask_gpu=worker_gpus[0],
        sequential=sequential,
        embed_gpu=embed,
        mask_gpus=list(worker_gpus),
    )


def resolve_assignment(
    *,
    cli_gpus: str | None = None,
    cli_mask_gpu: int | str | None = None,
    cli_embed_gpu: int | None = None,
    cli_mask_workers: int | None = None,
    config_devices=None,
    config_device: str | None = None,
    config_mask_device: str | None = None,
    config_mask_devices=None,
    config_embed_device: str | None = None,
    config_mask_workers: int | None = None,
) -> GpuAssignment:
    """CLI ``--gpus`` / ``--mask-gpu`` / ``--embed-gpu`` win over YAML devices.

    When ``--gpus`` is set, the reserved mask GPU defaults to the last id in that list
    (YAML ``masks.device`` is ignored unless ``--mask-gpu`` is also given). Embed GPU
    defaults to the first generation GPU. ``--mask-workers`` / YAML ``masks.workers``
    set how many blend processes to start. With neither CLI nor config devices, fall
    back to ``generation.device`` (default ``cuda:0``) as a single sequential GPU.
    """
    ids = parse_gpu_ids(cli_gpus)
    from_cli = ids is not None
    if ids is None:
        ids = parse_gpu_id_list(config_devices)
    if ids is None:
        fallback = parse_device_id(config_device) if config_device is not None else 0
        ids = [0 if fallback is None else fallback]
    if cli_mask_gpu is not None:
        mask = cli_mask_gpu
    elif from_cli:
        mask = None
    elif config_mask_devices is not None:
        mask = config_mask_devices
    else:
        mask = parse_device_id(config_mask_device)
    if cli_embed_gpu is not None:
        embed = cli_embed_gpu
    elif from_cli:
        embed = None
    else:
        embed = parse_device_id(config_embed_device)
    workers = (
        cli_mask_workers if cli_mask_workers is not None else config_mask_workers
    )
    if workers is not None:
        workers = int(workers)
    return assign_devices(ids, mask, embed, mask_workers=workers)


_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def cap_cpu_threads(n: int = 1) -> None:
    """Keep one BLAS/OMP thread per process so N GPU workers do not oversubscribe the CPU.

    Must run in the parent before ``Process.start`` so spawn children inherit it before
    they import numpy. Also disables tqdm / Hub progress bars so worker stdout cannot
    fill a piped log with ``\\r`` updates (that deadlocks a line-buffered reader).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for key in _THREAD_ENV:
        os.environ[key] = str(int(n))


def pin_visible_gpu(gpu: int) -> str:
    """Restrict this process to one physical GPU before CUDA is initialised.

    Diffusers pipeline ``from_pretrained(device_map=...)`` only accepts the strings
    ``balanced`` / ``cuda`` / ``cpu``, not a dict or ``cuda:N``. Setting
    ``CUDA_VISIBLE_DEVICES`` first makes ``device_map='cuda'`` (and ``.to('cuda')``) land
    on the intended card. Must be called before ``import torch`` / any CUDA init in this
    process. Returns the local device name ``cuda``.

    Also sets ``TORCH_DISABLE_NATIVE_JIT=1`` so PyTorch does not JIT-compile Triton
    kernels (Qwen3 RoPE hits ``bmm`` → Triton, which needs ``Python.h`` / python3.12-dev)
    and caps CPU thread pools so parallel workers do not fight over cores.
    """
    cap_cpu_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"
    return "cuda"


@contextmanager
def hide_parent_cuda() -> Iterator[None]:
    """Blank ``CUDA_VISIBLE_DEVICES`` in this process so it cannot touch worker GPUs.

    Spawn children call :func:`pin_visible_gpu` and overwrite the variable before they
    import torch. Restored on exit so a later parent-side scorer can see the cards.
    """
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
