"""EditReward margin scoring for standalone generation.

Gates on ``score(clean, edit, prompt) − score(clean, clean, prompt)``. Heavy
``EditReward`` import is lazy so the module stays cheap without the checkpoint loaded.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RewardVerdict:
    """One EditReward margin decision for a candidate."""

    decision: str  # "accept" | "reject"
    margin: float
    edit_score: float
    noop_score: float


_TEXT_CONFIG_ALIASES = (
    "hidden_size",
    "pad_token_id",
    "vocab_size",
    "use_cache",
    "rms_norm_eps",
)


def _patch_transformers_trainer_for_editreward() -> None:
    """Make ``import EditReward`` work on transformers 5.x without pinning 4.x.

    EditReward's trainer module does ``from transformers.trainer import …,
    DistributedTensorGatherer, SequentialDistributedSampler, nested_concat``.
    Transformers 5 dropped the first two (TPU eval helpers unused even in
    EditReward's trainer body) and moved ``nested_concat`` to
    ``trainer_pt_utils``. Inference never calls any of them — the import is
    the only load-bearing line, so stub / re-export the missing names instead of
    downgrading transformers.
    """
    import transformers.trainer as trainer

    if not hasattr(trainer, "nested_concat"):
        from transformers.trainer_pt_utils import nested_concat

        trainer.nested_concat = nested_concat
    if not hasattr(trainer, "DistributedTensorGatherer"):
        trainer.DistributedTensorGatherer = type("DistributedTensorGatherer", (), {})
    if not hasattr(trainer, "SequentialDistributedSampler"):
        trainer.SequentialDistributedSampler = type("SequentialDistributedSampler", (), {})


def _flatten_vl_text_config(config: Any) -> None:
    """Copy nested ``text_config`` fields EditReward still reads off the parent config.

    Transformers 5's Qwen2.5-VL config has no top-level ``hidden_size`` /
    ``pad_token_id`` / ``use_cache``; they live on ``text_config``. The reward
    head is ``nn.Linear(config.hidden_size, …)``.
    """
    text = getattr(config, "text_config", None)
    if text is None:
        return
    for name in _TEXT_CONFIG_ALIASES:
        if getattr(config, name, None) is not None:
            continue
        value = getattr(text, name, None)
        if value is not None:
            setattr(config, name, value)


def _filter_reward_init_kwargs(init: Any, config: Any, kwargs: dict) -> dict:
    """Keep kwargs the reward-model ``__init__`` accepts; stash ``use_cache`` on config.

    Transformers 5's ``from_pretrained`` does ``cls(config, **leftover)``. EditReward
    itself passes ``use_cache=False``, which 4.x absorbed onto the config and 5.x
    forwards into ``__init__``, whose closed signature then raises.
    """
    if "use_cache" in kwargs:
        config.use_cache = kwargs["use_cache"]
        text = getattr(config, "text_config", None)
        if text is not None:
            text.use_cache = kwargs["use_cache"]
    accepted = set(inspect.signature(init).parameters) - {"self", "kwargs"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _remap_qwen25_vl_state_dict(state_dict: dict) -> dict:
    """Map transformers 4.x Qwen2.5-VL keys onto the 5.x nested layout."""
    keys = list(state_dict)
    if any(k.startswith("model.language_model.") for k in keys):
        return state_dict
    remapped: dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("model.") and not key.startswith(
            ("model.visual.", "model.language_model.")
        ):
            rest = key[len("model.") :]
            if rest.split(".", 1)[0] in {"layers", "embed_tokens", "norm", "rotary_emb"}:
                new_key = "model.language_model." + rest
        remapped[new_key] = value
    return remapped


def _patch_reward_model_for_transformers5(cls: type) -> None:
    """Adapt EditReward's Qwen2.5-VL reward wrapper to transformers 5.x.

    * ``from_pretrained`` leftover kwargs (``use_cache``) no longer match ``__init__``.
    * ``hidden_size`` moved under ``text_config``.
    * Vision lives at ``model.visual``; 4.x ``self.visual(pixels)`` returned a tensor,
      5.x returns a ``ModelOutput`` and requires ``grid_thw``. Scoring therefore
      delegates vision+language to ``self.model`` (the combined VL body).
    * Fine-tune checkpoints may still use 4.x parameter names.
    """
    if getattr(cls.__init__, "_synevad_tf5_patched", False):
        return

    original_init = cls.__init__

    def patched_init(self, config, *args, **kwargs):
        _flatten_vl_text_config(config)
        kwargs = _filter_reward_init_kwargs(original_init, config, kwargs)
        original_init(self, config, *args, **kwargs)

    patched_init._synevad_tf5_patched = True
    cls.__init__ = patched_init

    # 4.x code reads ``self.visual``; 5.x stores it at ``self.model.visual``.
    # Assigning ``self.visual = …`` registers a second nn.Module, so
    # ``state_dict()`` grows a ``visual.*`` prefix the checkpoint does not have
    # (``strict=True`` then fails). A property is not a registered child.
    if not isinstance(getattr(cls, "visual", None), property):
        cls.visual = property(lambda self: self.model.visual)

    original_lsd = cls.load_state_dict

    def patched_lsd(self, state_dict, *args, **kwargs):
        if isinstance(state_dict, dict):
            expected = self.state_dict().keys()
            remapped = _remap_qwen25_vl_state_dict(state_dict)
            if sum(k in expected for k in remapped) >= sum(k in expected for k in state_dict):
                state_dict = remapped
        return original_lsd(self, state_dict, *args, **kwargs)

    cls.load_state_dict = patched_lsd

    def run_batch_tf5(self, batch_dict, head_module):
        import torch

        input_ids = batch_dict.get("input_ids")
        inputs_embeds = batch_dict.get("inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided in batch_dict")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=batch_dict.get("attention_mask"),
            position_ids=batch_dict.get("position_ids"),
            past_key_values=batch_dict.get("past_key_values"),
            inputs_embeds=inputs_embeds,
            pixel_values=batch_dict.get("pixel_values"),
            pixel_values_videos=batch_dict.get("pixel_values_videos"),
            image_grid_thw=batch_dict.get("image_grid_thw"),
            video_grid_thw=batch_dict.get("video_grid_thw"),
            use_cache=batch_dict.get("use_cache") or False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        batch_size = input_ids.shape[0] if input_ids is not None else hidden_states.shape[0]
        device_type = "cuda" if hidden_states.is_cuda else hidden_states.device.type
        with torch.autocast(
            device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"
        ):
            logits = head_module(hidden_states)
        return self._pool_logits(logits, input_ids, batch_size)

    cls._run_single_batch_through_model_and_head = run_batch_tf5


def _disable_torch_native_triton() -> None:
    """Keep RoPE on eager ``aten::bmm`` instead of torch._native's Triton kernel.

    Transformers 5's Qwen2.5-VL RoPE does ``inv_freq @ position_ids`` (a batched
    outer product). New PyTorch routes that through a Triton JIT that compiles
    ``cuda_utils.c``, which needs ``Python.h`` (the ``python3.x-dev`` headers).
    This box does not have them, so the first ``reward()`` dies in gcc. Scoring
    does not need that kernel; disable it before the model runs.
    """
    import os
    import sys

    os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"
    native = sys.modules.get("torch._native")
    if native is None:
        return
    native.triton_utils.deregister_op_overrides()


def load_inferencer(
    *,
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda:0",
):
    """Construct an ``EditRewardInferencer`` (requires the EditReward package)."""
    _disable_torch_native_triton()
    _patch_transformers_trainer_for_editreward()
    from EditReward import EditRewardInferencer
    from EditReward.model.qwen2_5_vl_trainer import Qwen2_5_VLRewardModelBT_MultiHead

    _patch_reward_model_for_transformers5(Qwen2_5_VLRewardModelBT_MultiHead)

    return EditRewardInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        reward_dim="overall_detail",
        rm_head_type="ranknet_multi_head",
        device=device,
    )


def _as_float(score: Any) -> float:
    """Coerce a reward() return value (tensor / list / scalar) to float."""
    if hasattr(score, "detach"):
        score = score.detach().cpu()
    if hasattr(score, "tolist"):
        score = score.tolist()
    if isinstance(score, (list, tuple)):
        score = score[0]
        if isinstance(score, (list, tuple)):
            score = score[0]
    return float(score)


def score_margin(
    inferencer,
    clean: Image.Image,
    edit: Image.Image,
    prompt: str,
    *,
    noop_cache: dict[tuple[str, str], float] | None = None,
    cache_key: tuple[str, str] | None = None,
    accept_margin: float = 0.5,
) -> RewardVerdict:
    """Score one (clean, edit, prompt) triple; accept when margin >= accept_margin."""
    if noop_cache is not None and cache_key is not None and cache_key in noop_cache:
        noop = noop_cache[cache_key]
    else:
        noop = _as_float(inferencer.reward([prompt], [clean], [clean])[0])
        if noop_cache is not None and cache_key is not None:
            noop_cache[cache_key] = noop
    edit_score = _as_float(inferencer.reward([prompt], [clean], [edit])[0])
    margin = edit_score - noop
    return RewardVerdict(
        decision="accept" if margin >= accept_margin else "reject",
        margin=margin,
        edit_score=edit_score,
        noop_score=noop,
    )
