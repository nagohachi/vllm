# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Local fork addition; not for upstream submission. Mirrors the upstream
# Qwen2-Audio model wiring for our `custom_lalm` architecture
# (WhisperEncoder + stack-MLP projector + Qwen3 LM). The HF reference is at
# libs/ms-swift/examples/custom/custom_lalm/.
"""Inference-only CustomLALM model (audio-only LALM) compatible with HF weights.

Architecture (mirrors `CustomLALMPretrainedModel` from the HF side):
- `audio_tower`   : `transformers.WhisperEncoder` over a 30 s mel-padded input.
- `audio_projector`: stack-`k` + 2-layer MLP (silu); identical state-dict shape
  to the HF version, so `AutoWeightsLoader` lines up.
- `language_model`: Qwen3 LM, instantiated via vLLM's registry so KV cache,
  paged attention, and pipeline-parallel come for free.

Token contract: each `<|audio_pad|>` placeholder is expanded by the multimodal
processor into `K_audio = ceil((real_mel_frames // 2) / audio_stack_factor)`
copies, one per output token. The model's `embed_multimodal` produces exactly
that many vectors per clip; vLLM scatters them into `inputs_embeds` at the
placeholder positions before the LM forward.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


# === Constants ===
AUDIO_TOKEN = "<|audio_pad|>"


# === Audio Inputs (model-side) ===
class CustomLALMAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins (Whisper-medium: 80)
        - T: Number of mel frames (Whisper default chunk: 3000)
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nmb", "T"),
    ]
    audio_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", "T"),
    ]


CustomLALMAudioInputs: TypeAlias = CustomLALMAudioFeatureInputs


# === Audio projector (matches HF CustomLALMProjector) ===
class CustomLALMAudioProjector(nn.Module):
    """Stack `stack_factor` encoder frames, then MLP-project to LM hidden dim.

    State-dict layout (`mlp.0.{weight,bias}` and `mlp.2.{weight,bias}`) is
    identical to the HF version so checkpoint loading is direct.
    """

    def __init__(
        self,
        audio_hidden_size: int,
        stack_factor: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        assert stack_factor >= 1
        self.stack_factor = stack_factor
        stacked_dim = audio_hidden_size * stack_factor
        self.mlp = nn.Sequential(
            nn.Linear(stacked_dim, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.stack_factor
        if k > 1:
            t = x.shape[1]
            pad = (-t) % k
            if pad:
                x = F.pad(x, (0, 0, 0, pad))
                attention_mask = F.pad(attention_mask, (0, pad))
            x = rearrange(x, "n (t k) d -> n t (k d)", k=k)
            attention_mask = (
                rearrange(attention_mask, "n (t k) -> n t k", k=k).any(dim=-1).long()
            )
        return self.mlp(x), attention_mask


# === Helpers shared with the multimodal processor ===
def _whisper_encoder_output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
    """Whisper conv stride 2 (conv2): T_out = T_in // 2."""
    return mel_lengths // 2


def _custom_lalm_audio_token_count(
    mel_lengths: torch.Tensor, stack_factor: int
) -> torch.Tensor:
    """`K_audio = ceil((mel_frames // 2) / stack_factor)` per clip."""
    encoder_lengths = _whisper_encoder_output_lengths(mel_lengths)
    return (encoder_lengths + stack_factor - 1) // stack_factor


def _custom_lalm_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(
        input_features=MultiModalFieldConfig.batched("audio"),
        audio_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


# === Processing info / dummy inputs / processor ===
class CustomLALMProcessingInfo(BaseProcessingInfo):
    """Tells vLLM about the HF processor, the feature extractor, and the
    max-tokens budget per audio clip."""

    def get_hf_config(self):
        # The HF config is loaded with `trust_remote_code=True` and carries our
        # nested `audio_config` / `lm_config`. Return it untyped — `ctx.get_hf_config`
        # would otherwise demand a built-in config class.
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        # Pass no `typ` arg → vLLM's loader takes the `ProcessorMixin` branch,
        # which dispatches through `AutoProcessor.from_pretrained` and resolves
        # our `CustomLALMProcessor` via `processor_config.json`'s `auto_map`.
        # Passing `AutoProcessor` explicitly hits the wrong branch (it would
        # try `AutoProcessor(**kwargs)` directly → OSError).
        return self.ctx.get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        # Our `CustomLALMProcessor` exposes the Whisper feature extractor as
        # `audio_processor` (different attribute name from Qwen2-Audio's
        # `feature_extractor`).
        feature_extractor = hf_processor.audio_processor
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        mm_counts = mm_counts or {}
        if mm_counts.get("audio", 0) <= 0:
            return {}

        feature_extractor = self.get_feature_extractor()
        chunk_length = min(feature_extractor.chunk_length, 30)
        audio_len = int(chunk_length * feature_extractor.sampling_rate)
        hop_length = feature_extractor.hop_length
        max_mel_seq_len = audio_len // hop_length  # 3000 by default

        stack_factor = int(getattr(self.get_hf_processor(), "audio_stack_factor", 4))
        max_tokens = _custom_lalm_audio_token_count(
            torch.tensor([max_mel_seq_len], dtype=torch.long),
            stack_factor,
        )
        return {"audio": int(max_tokens.item())}


class CustomLALMDummyInputsBuilder(BaseDummyInputsBuilder[CustomLALMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Bare `<|audio_pad|>` — the processor expands it. We don't include the
        # `<|audio_start|>`/`<|audio_end|>` wrapper here because that's a chat-
        # template concern, not a placeholder-replacement concern.
        return AUDIO_TOKEN * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class CustomLALMMultiModalProcessor(BaseMultiModalProcessor[CustomLALMProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # vLLM uses "audio" as the data key; our `CustomLALMProcessor.__call__`
        # takes `audio=...` too, so just forward.
        audios = mm_data.get("audio", []) or mm_data.get("audios", [])
        if not audios:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Normalise: our processor signature is `audio=...` (singular), matching
        # the HF Qwen-VL/Omni convention.
        mm_data = dict(mm_data)
        if "audios" in mm_data and "audio" not in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _custom_lalm_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = hf_processor.tokenizer
        audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
        stack_factor = int(getattr(hf_processor, "audio_stack_factor", 4))

        out_mm_data = out_mm_kwargs.get_data()
        audio_attention_mask = out_mm_data.get("audio_attention_mask")
        if audio_attention_mask is None:
            audio_output_lengths: list[int] = []
        else:
            assert isinstance(audio_attention_mask, torch.Tensor)
            mel_lengths = audio_attention_mask.sum(-1)
            audio_output_lengths = _custom_lalm_audio_token_count(
                mel_lengths, stack_factor
            ).tolist()

        def get_replacement(item_idx: int) -> list[int]:
            if not audio_output_lengths:
                raise RuntimeError(
                    "audio_output_lengths missing — processor didn't emit a mask"
                )
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                raise ValueError(
                    f"Audio at index {item_idx} (len={audio_len}) is too short "
                    "to fit a single audio token after stack_factor reduction."
                )
            return [audio_token_id] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]


# === Main model ===
@MULTIMODAL_REGISTRY.register_processor(
    CustomLALMMultiModalProcessor,
    info=CustomLALMProcessingInfo,
    dummy_inputs=CustomLALMDummyInputsBuilder,
)
class CustomLALMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            # Match the bracketing the chat template injects so user-supplied
            # prompts that bypass `apply_chat_template` still render audio
            # consistently with training prompts.
            return f"<|audio_start|>{AUDIO_TOKEN}<|audio_end|>"
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        audio_config = config.audio_config
        lm_config = config.lm_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = WhisperEncoder(audio_config)
            self.audio_projector = CustomLALMAudioProjector(
                audio_hidden_size=audio_config.d_model,
                stack_factor=int(getattr(config, "audio_stack_factor", 4)),
                hidden_size=lm_config.hidden_size,
                output_size=lm_config.hidden_size,
            )

        # Hand the LM to vLLM's registry — it picks the right vLLM-side Qwen3
        # impl (paged attn, KV cache, PP/TP), then `AutoWeightsLoader` routes
        # the `language_model.*` state dict here.
        lm_archs = getattr(lm_config, "architectures", None) or ["Qwen3ForCausalLM"]
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=lm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=lm_archs,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> CustomLALMAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        if input_features is None:
            return None
        return CustomLALMAudioFeatureInputs(
            type="audio_features",
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
        )

    def _process_audio_input(
        self, audio_input: CustomLALMAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        input_features = audio_input["input_features"]
        audio_attention_mask = audio_input["audio_attention_mask"]

        # Dtype align: feature extractor emits fp32; tower weights are model-dtype.
        tower_dtype = self.audio_tower.conv1.weight.dtype
        if input_features.dtype != tower_dtype:
            input_features = input_features.to(tower_dtype)

        # WhisperEncoder: (N, n_mels, T_in) -> (N, T_out = T_in // 2, d_model)
        encoder_out = self.audio_tower(
            input_features=input_features, return_dict=True
        )
        hidden = encoder_out.last_hidden_state
        t_out = hidden.shape[1]

        # Pool the mel mask the same way conv2 (stride 2) collapses time.
        mel_mask = audio_attention_mask
        if mel_mask.shape[1] < t_out * 2:
            mel_mask = F.pad(mel_mask, (0, t_out * 2 - mel_mask.shape[1]))
        elif mel_mask.shape[1] > t_out * 2:
            mel_mask = mel_mask[:, : t_out * 2]
        out_mask = rearrange(mel_mask, "n (t k) -> n t k", k=2).any(dim=-1).long()

        proj_dtype = next(self.audio_projector.parameters()).dtype
        if hidden.dtype != proj_dtype:
            hidden = hidden.to(proj_dtype)
        projected, downsampled_mask = self.audio_projector(hidden, out_mask)

        # Per-clip valid token counts, then flatten + split back per clip.
        per_clip_counts = downsampled_mask.sum(dim=-1).tolist()
        flat = projected[downsampled_mask.bool()]
        return tuple(torch.split(flat, per_clip_counts))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
