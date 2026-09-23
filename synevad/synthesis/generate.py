"""FLUX.2 editing: pipeline load + a single-crop img2img or inpaint call.

:func:`load_pipeline` constructs a Klein or full FLUX.2 pipeline (optionally 8-bit), or the
Klein inpainting pipeline when ``task="inpaint"``.
:func:`edit_image` renders one crop -- square unless ``resize_to`` is a ``(w, h)`` pair.
:func:`inpaint_image` regenerates only the white pixels of a mask (negative controls).
``torch`` / ``diffusers`` are imported lazily so this module stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
_EIGHT_BIT = frozenset({"8bit", "8-bit", "int8"})
_NONE = frozenset({"", "none", "bf16", "bfloat16"})


def uses_klein(base_model: str) -> bool:
    """True when ``base_model`` is a FLUX.2 Klein checkpoint."""
    return "klein" in str(base_model).lower()


def normalize_quantization(quantization: str | None) -> str:
    """Map config/CLI aliases onto ``8bit`` or ``none``."""
    raw = (quantization or "none").strip().lower()
    if raw in _EIGHT_BIT:
        return "8bit"
    if raw in _NONE:
        return "none"
    raise ValueError(
        f"unknown quantization {quantization!r}; expected '8bit' or 'none'"
    )


TASKS = ("edit", "inpaint")


def pipeline_class_name(base_model: str, task: str = "edit") -> str:
    """The diffusers pipeline class for ``base_model`` and ``task``.

    Inpainting exists for Klein only (``Flux2KleinInpaintPipeline``); asking for it on full
    FLUX.2 fails here rather than silently falling back to whole-crop editing.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}, got {task!r}")
    if task == "inpaint":
        if not uses_klein(base_model):
            raise ValueError(
                f"inpainting needs a FLUX.2 Klein checkpoint; {base_model!r} has no inpaint pipeline"
            )
        return "Flux2KleinInpaintPipeline"
    return "Flux2KleinPipeline" if uses_klein(base_model) else "Flux2Pipeline"


@dataclass(frozen=True)
class PipelineSpec:
    """How :func:`load_pipeline` will construct the editor (no torch)."""

    pipeline_cls: str
    torch_dtype: str
    quantization: str
    device: str
    device_map: str | None
    cpu_offload: bool
    quant_backend: str | None
    quant_kwargs: dict[str, Any] | None
    components_to_quantize: tuple[str, ...] | None
    # Per-component bitsandbytes: ("module", "diffusers"|"transformers", kwargs)
    quant_mapping: tuple[tuple[str, str, dict[str, Any]], ...] | None
    include_text_encoder: bool


def _nf4_4bit() -> dict[str, Any]:
    # fp16 compute: Quadro RTX 6000 is Turing (no bf16 tensor cores). The torch
    # profile spent ~87% of CUDA time in magma_sgemmEx on bf16 4-bit GEMM.
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
    }


def _transformer_4bit() -> tuple[str, str, dict[str, Any]]:
    return ("transformer", "diffusers", _nf4_4bit())


def _text_encoder_4bit() -> tuple[str, str, dict[str, Any]]:
    return ("text_encoder", "transformers", _nf4_4bit())


def pipeline_spec(
    base_model: str,
    *,
    device: str = "cuda:0",
    quantization: str | None = "none",
    cpu_offload: bool = False,
    include_text_encoder: bool = True,
    task: str = "edit",
) -> PipelineSpec:
    """Describe the editor load without importing torch or diffusers.

    Diffusers pipeline ``device_map`` only accepts the strings ``balanced`` / ``cuda`` /
    ``cpu``. 8-bit uses ``device_map='cuda'`` after :func:`synevad.synthesis.devices.pin_visible_gpu`.

    Klein 9B is a 9B DiT **plus** Qwen3-8B. The standalone runner encodes prompts on a
    separate GPU and loads FLUX workers with ``include_text_encoder=False`` (DiT only).
    The 8-bit preset is a 24 GB recipe: DiT 4-bit NF4 (~4.5 GB) + Qwen3 4-bit NF4
    (~5 GB). LLM.int8 on the DiT still left ~18 GB of BF16 weights, and native SDPA
    then tried to allocate a 9 GB attention matrix.
    """
    quant = normalize_quantization(quantization)
    cls = pipeline_class_name(base_model, task)
    keep_te = bool(include_text_encoder)
    if quant == "8bit":
        mapping = (_transformer_4bit(),)
        if keep_te:
            mapping = mapping + (_text_encoder_4bit(),)
        names = tuple(row[0] for row in mapping)
        return PipelineSpec(
            pipeline_cls=cls,
            torch_dtype="bfloat16",
            quantization=quant,
            device=device,
            device_map="cuda",
            cpu_offload=False,
            quant_backend=None,
            quant_kwargs=None,
            components_to_quantize=names,
            quant_mapping=mapping,
            include_text_encoder=keep_te,
        )
    return PipelineSpec(
        pipeline_cls=cls,
        torch_dtype="bfloat16",
        quantization=quant,
        device=device,
        device_map=None,
        cpu_offload=bool(cpu_offload),
        quant_backend=None,
        quant_kwargs=None,
        components_to_quantize=None,
        quant_mapping=None,
        include_text_encoder=keep_te,
    )


def _profile_attn() -> bool:
    from synevad.synthesis.profile import enabled

    return enabled()


def _disable_native_jit() -> None:
    import os

    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


def _cap_torch_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _log_cuda_identity(role: str) -> None:
    """Prove this process sees exactly one GPU after :func:`pin_visible_gpu`."""
    import os
    import torch

    if not torch.cuda.is_available():
        return
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    n = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if n else None
    name = getattr(props, "name", "cpu")
    uuid = getattr(props, "uuid", None)
    print(
        f"[{role}] pid={os.getpid()} CUDA_VISIBLE_DEVICES={vis!r} "
        f"device_count={n} name={name} uuid={uuid}",
        flush=True,
    )
    if n != 1:
        print(
            f"[{role}] WARNING: expected 1 visible GPU after pin; "
            "workers may be sharing a card",
            flush=True,
        )


def _strip_accelerate_hooks(pipeline) -> None:
    """Drop device-map hooks so each ``__call__`` does not re-dispatch / CPU-roundtrip."""
    try:
        from accelerate.hooks import remove_hook_from_module
        import torch.nn as nn
    except Exception:
        return
    for name in ("transformer", "vae", "text_encoder"):
        mod = getattr(pipeline, name, None)
        if isinstance(mod, nn.Module):
            try:
                remove_hook_from_module(mod, recurse=True)
            except Exception:
                pass
    if getattr(pipeline, "hf_device_map", None):
        pipeline.hf_device_map = None


def _deregister_triton() -> None:
    try:
        from torch._native.triton_utils import deregister_op_overrides

        deregister_op_overrides()
    except Exception:
        pass


def _log_transformer_memory(pipeline) -> None:
    """Print quantized-linear counts and param bytes so a 24 GB OOM is diagnosable."""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return
    n_8 = n_4 = n_lin = 0
    for module in transformer.modules():
        name = type(module).__name__
        lower = name.lower()
        if "8bit" in lower:
            n_8 += 1
        elif "4bit" in lower:
            n_4 += 1
        elif name == "Linear":
            n_lin += 1
    param_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters())
    extra = ""
    try:
        import torch

        if torch.cuda.is_available():
            extra = f" cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f}G"
    except Exception:
        pass
    print(
        f"[gen] transformer Linear8bitLt={n_8} Linear4bit={n_4} Linear={n_lin} "
        f"param_bytes={param_bytes / 1e9:.2f}G{extra}",
        flush=True,
    )


def _chunked_sdpa(query, key, value, *, attn_mask, dropout_p, is_causal, scale, chunk: int = 256):
    """Math SDPA in query-sequence chunks so the score matrix is chunk×S, not S×S."""
    import torch
    import torch.nn.functional as F

    seq = query.shape[2]
    parts = []
    for start in range(0, seq, chunk):
        q = query[:, :, start : start + chunk]
        mask = attn_mask
        if mask is not None and mask.ndim >= 2:
            mask = mask[..., start : start + chunk, :]
        parts.append(
            F.scaled_dot_product_attention(
                q, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
        )
    return torch.cat(parts, dim=2)


def _fp16_sdpa(
    query,
    key,
    value,
    *,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
):
    """SDPA on ``(B, S, H, D)`` tensors without an fp32 S×S score matrix.

    This torch build's mem-efficient kernel rejects bfloat16 (Half/Float only) and
    math SDPA upcasts bf16 to fp32 (~9 GiB at 1024² img2img). Cast to fp16 and use
    mem-efficient; fall back to chunked math.
    """
    import time

    import torch
    import torch.nn.functional as F

    orig_dtype = query.dtype
    t_cast0 = time.perf_counter() if _profile_attn() else None
    query_h = query.permute(0, 2, 1, 3).contiguous().to(torch.float16)
    key_h = key.permute(0, 2, 1, 3).contiguous().to(torch.float16)
    value_h = value.permute(0, 2, 1, 3).contiguous().to(torch.float16)
    mask = None if attn_mask is None else attn_mask.to(dtype=torch.float16)
    if t_cast0 is not None:
        from synevad.synthesis.profile import attn_add, cuda_sync_enabled

        if cuda_sync_enabled():
            torch.cuda.synchronize()
        attn_add(time.perf_counter() - t_cast0, part="cast")
    kwargs: dict[str, Any] = {
        "attn_mask": mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
    }
    chunked = False
    t0 = time.perf_counter() if _profile_attn() else None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(query_h, key_h, value_h, **kwargs)
    except Exception:
        chunked = True
        out = _chunked_sdpa(
            query_h,
            key_h,
            value_h,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
    if t0 is not None:
        from synevad.synthesis.profile import attn_add, cuda_sync_enabled

        if cuda_sync_enabled():
            torch.cuda.synchronize()
        attn_add(time.perf_counter() - t0, chunked=chunked, part="sdpa")
        t_back = time.perf_counter()
        out = out.permute(0, 2, 1, 3).to(orig_dtype)
        if cuda_sync_enabled():
            torch.cuda.synchronize()
        attn_add(time.perf_counter() - t_back, part="cast")
        return out
    return out.permute(0, 2, 1, 3).to(orig_dtype)


def _low_mem_attention_dispatch(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
    attention_kwargs=None,
    *,
    backend=None,
    parallel_config=None,
):
    """Drop-in for ``dispatch_attention_fn`` used by Flux2 attention processors.

    Routes every Flux2 block through the fp16 mem-efficient kernel below, so the DiT never
    materialises a 9 GiB fp32 score matrix.
    """
    return _fp16_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _install_low_mem_attention() -> None:
    """Patch Flux2's imported ``dispatch_attention_fn`` in this process."""
    from diffusers.models.transformers import transformer_flux2

    if getattr(transformer_flux2, "dispatch_attention_fn", None) is _low_mem_attention_dispatch:
        return
    transformer_flux2.dispatch_attention_fn = _low_mem_attention_dispatch
    print("[gen] attention: fp16 mem-efficient SDPA (no 9 GiB fp32 scores)", flush=True)


def load_pipeline(
    base_model: str,
    *,
    device: str = "cuda:0",
    quantization: str | None = "none",
    cpu_offload: bool = False,
    include_text_encoder: bool = True,
    task: str = "edit",
):
    """Load a FLUX.2 (Klein or full) img2img pipeline onto ``device``.

    When ``include_text_encoder`` is false the Qwen weights are not loaded; the caller
    must pass ``prompt_embeds`` into :func:`edit_image`. ``task="inpaint"`` loads the Klein
    inpainting pipeline from the same checkpoint instead.
    """
    _disable_native_jit()
    import torch

    _cap_torch_threads()
    _deregister_triton()
    _log_cuda_identity("gen")
    import diffusers
    from diffusers import (
        BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
        PipelineQuantizationConfig,
    )
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    spec = pipeline_spec(
        base_model,
        device=device,
        quantization=quantization,
        cpu_offload=cpu_offload,
        include_text_encoder=include_text_encoder,
        task=task,
    )
    cls = getattr(diffusers, spec.pipeline_cls)
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if spec.quant_mapping:
        mapping: dict[str, Any] = {}
        for name, origin, qkwargs in spec.quant_mapping:
            raw = dict(qkwargs)
            if raw.get("bnb_4bit_compute_dtype") == "bfloat16":
                raw["bnb_4bit_compute_dtype"] = torch.bfloat16
            elif raw.get("bnb_4bit_compute_dtype") == "float16":
                raw["bnb_4bit_compute_dtype"] = torch.float16
            cfg_cls = (
                DiffusersBitsAndBytesConfig
                if origin == "diffusers"
                else TransformersBitsAndBytesConfig
            )
            mapping[name] = cfg_cls(**raw)
        kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=mapping)
        kwargs["device_map"] = spec.device_map
    elif spec.quant_backend is not None:
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=spec.quant_backend,
            quant_kwargs=dict(spec.quant_kwargs or {}),
            components_to_quantize=list(spec.components_to_quantize or []),
        )
        kwargs["device_map"] = spec.device_map
    if not spec.include_text_encoder:
        kwargs["text_encoder"] = None
    from synevad.synthesis.profile import span as _span

    with _span("load_pipeline.from_pretrained", model=base_model):
        pipeline = cls.from_pretrained(base_model, **kwargs)
    _strip_accelerate_hooks(pipeline)
    if spec.device_map is None:
        if spec.cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "config") and getattr(pipeline.vae.config, "force_upcast", False):
            pipeline.vae.config.force_upcast = False
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
    if getattr(pipeline, "transformer", None) is not None:
        _install_low_mem_attention()
    _log_transformer_memory(pipeline)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def load_text_encoder(base_model: str, *, quantization: str | None = "none"):
    """Load Klein's Qwen3 text encoder + tokenizer (no DiT / VAE).

    Call after :func:`synevad.synthesis.devices.pin_visible_gpu` so ``device_map='cuda'`` lands on
    the embed card. 8-bit means 4-bit NF4 on this encoder.
    """
    _disable_native_jit()
    import torch

    _cap_torch_threads()
    _deregister_triton()
    _log_cuda_identity("embed")
    from transformers import BitsAndBytesConfig, Qwen2TokenizerFast, Qwen3ForCausalLM

    tokenizer = Qwen2TokenizerFast.from_pretrained(base_model, subfolder="tokenizer")
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if normalize_quantization(quantization) == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    from synevad.synthesis.profile import span as _span

    with _span("load_text_encoder.from_pretrained", model=base_model):
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model, subfolder="text_encoder", **kwargs
        )
    return tokenizer, text_encoder


def encode_prompt_texts(tokenizer, text_encoder, prompts: list[str]) -> dict[str, Any]:
    """Map each prompt string to its CPU Klein prompt-embed tensor.

    Encoded once on the embed GPU and written to disk, so the DiT workers can load with
    ``text_encoder=None`` and never pay for Qwen3.
    """
    from diffusers import Flux2KleinPipeline

    out: dict[str, Any] = {}
    for text in prompts:
        embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=text,
        )
        out[text] = embeds.detach().cpu().contiguous()
    return out


def save_prompt_embeds(path, mapping: dict[str, Any]) -> None:
    import torch
    from pathlib import Path

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(mapping, tmp)
    tmp.replace(dest)


def load_prompt_embeds(path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def edit_image(
    pipe,
    clean_crop,
    prompt,
    *,
    steps,
    guidance,
    seed,
    resize_to,
    prompt_embeds=None,
):
    """Edit one crop with the FLUX.2 pipeline; returns a PIL image.

    ``clean_crop`` is a PIL image already sized to ``resize_to``, which is a square side
    or a ``(w, h)`` pair. The generator is seeded on CPU for reproducibility. Pass
    ``prompt_embeds`` (from the embed GPU) to skip Qwen; ``prompt`` is then ignored.
    """
    import inspect
    import time

    import torch

    from synevad.synthesis.crops import resize_size

    from synevad.synthesis.profile import (
        attn_flush,
        cuda_mem,
        emit,
        enabled as profile_on,
        maybe_torch_profile,
        span,
    )

    width, height = resize_size(resize_to)
    call: dict[str, Any] = {
        "image": clean_crop,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cpu").manual_seed(int(seed)),
    }
    if prompt_embeds is not None:
        device = getattr(pipe, "_execution_device", None) or "cuda"
        call["prompt_embeds"] = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    else:
        call["prompt"] = prompt

    step_t0 = time.perf_counter()
    attn_flush()

    def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        nonlocal step_t0
        if profile_on():
            extra = attn_flush()
            mem = cuda_mem()
            if mem:
                extra.update(mem)
            emit(
                "dit_step",
                dur_ms=(time.perf_counter() - step_t0) * 1000.0,
                step=int(step_index),
                **extra,
            )
            step_t0 = time.perf_counter()
        return callback_kwargs

    if profile_on():
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        if "callback_on_step_end" in params or not params:
            call["callback_on_step_end"] = _on_step_end

    with span("edit_image", steps=int(steps)):
        with maybe_torch_profile():
            with torch.inference_mode():
                out = pipe(**call)
    return out.images[0]


def inpaint_image(
    pipe,
    clean_crop,
    mask,
    prompt,
    *,
    steps,
    guidance,
    seed,
    resize_to,
    strength: float = 1.0,
    prompt_embeds=None,
):
    """Regenerate the white pixels of ``mask`` in one crop with the Klein inpaint pipeline.

    ``mask`` is a PIL ``L`` image the size of ``clean_crop`` (white = repaint). ``strength``
    1.0 regenerates the masked area from noise, so the model cannot simply keep the original
    pixels there. Seeding and ``prompt_embeds`` behave as in :func:`edit_image`.
    """
    import torch

    from synevad.synthesis.crops import resize_size
    from synevad.synthesis.profile import span

    width, height = resize_size(resize_to)
    call: dict[str, Any] = {
        "image": clean_crop,
        "mask_image": mask,
        "strength": float(strength),
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device="cpu").manual_seed(int(seed)),
    }
    if prompt_embeds is not None:
        device = getattr(pipe, "_execution_device", None) or "cuda"
        call["prompt_embeds"] = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    else:
        call["prompt"] = prompt
    with span("inpaint_image", steps=int(steps)):
        with torch.inference_mode():
            out = pipe(**call)
    return out.images[0]
