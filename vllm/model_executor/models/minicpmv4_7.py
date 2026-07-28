# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-V OCR 4.7 model."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    ResolvedPromptUpdate,
)

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsLoRA,
)
from .minicpmv import (
    MiniCPMVBaseModel,
    MiniCPMVDummyInputsBuilder,
    MiniCPMVImageEmbeddingItems,
    MiniCPMVImageInputs,
    MiniCPMVImagePixelInputs,
    MiniCPMVMultiModalProcessor,
    MiniCPMVProcessingInfo,
)
from .module_mapping import MultiModelKeys
from .qwen3_5 import Qwen3_5ForCausalLM
from .utils import AutoWeightsLoader

_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_LEGACY_IMAGE_PLACEHOLDER = "(<image>./</image>)"


def _minicpmv4_7_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    fields = {
        "pixel_values": MultiModalFieldConfig.batched("image"),
        "image_sizes": MultiModalFieldConfig.batched("image"),
        "tgt_sizes": MultiModalFieldConfig.batched("image"),
        "grids": MultiModalFieldConfig.batched("image"),
        "source_image_visual_tokens": MultiModalFieldConfig.batched("image"),
        "patch_visual_tokens": MultiModalFieldConfig.batched("image"),
        "image_embeds": MultiModalFieldConfig.batched("image"),
    }
    if "use_vit_merger" in hf_inputs:
        fields["use_vit_merger"] = MultiModalFieldConfig.batched("image")
    return fields


class MiniCPMV4_7ProcessingInfo(MiniCPMVProcessingInfo):
    image_pattern = _LEGACY_IMAGE_PLACEHOLDER

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(**kwargs)
        image_processor = hf_processor.image_processor
        for attr in ("mean", "std", "image_mean", "image_std"):
            value = getattr(image_processor, attr, None)
            if isinstance(value, np.ndarray):
                setattr(image_processor, attr, value.tolist())
        return hf_processor

    def get_model_version(self):
        return (4, 7)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_max_slice_num(self) -> int:
        config = self.get_hf_config()
        slice_config = getattr(config, "slice_config", None)
        if slice_config is not None:
            return getattr(slice_config, "max_slice_nums", 9)
        return getattr(config, "max_slice_nums", 9)

    def _get_downsample_mode(
        self,
        downsample_mode: str | None = None,
    ) -> str:
        if downsample_mode is not None:
            return downsample_mode
        image_processor = self.get_image_processor()
        return getattr(image_processor, "downsample_mode", "16x")

    def _compute_visual_tokens(
        self,
        image_size,
        max_slice_nums: int | None = None,
        downsample_mode: str | None = None,
    ) -> tuple[list[int], int, int]:
        image_processor = self.get_image_processor()
        if max_slice_nums is None:
            max_slice_nums = image_processor.max_slice_nums

        size = (image_size.width, image_size.height)
        grid = image_processor.get_sliced_grid(size, max_slice_nums)
        patch_size = image_processor.patch_size
        scale_resolution = image_processor.scale_resolution
        token_divisor = 4 if self._get_downsample_mode(downsample_mode) == "4x" else 16

        best_size = image_processor.find_best_resize(
            size,
            scale_resolution,
            patch_size,
            allow_upscale=grid is None,
        )
        source_tokens = (
            best_size[0] * best_size[1] // (patch_size * patch_size * token_divisor)
        )
        if grid is None:
            return [0, 0], source_tokens, source_tokens

        refine_size = image_processor.get_refine_size(
            size,
            grid,
            scale_resolution,
            patch_size,
            allow_upscale=True,
        )
        patch_width = refine_size[0] // grid[0]
        patch_height = refine_size[1] // grid[1]
        patch_tokens = (
            patch_width * patch_height // (patch_size * patch_size * token_divisor)
        )
        return list(grid), source_tokens, patch_tokens

    def get_slice_image_placeholder(
        self,
        image_size,
        image_idx: int = 0,
        max_slice_nums: int | None = None,
        use_image_id: bool = True,
        downsample_mode: str | None = None,
    ) -> str:
        grid, source_tokens, patch_tokens = self._compute_visual_tokens(
            image_size,
            max_slice_nums=max_slice_nums,
            downsample_mode=downsample_mode,
        )
        return self.get_image_processor().get_slice_image_placeholder(
            grid,
            image_idx=image_idx,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            source_image_visual_tokens=source_tokens,
            patch_visual_tokens=patch_tokens,
        )

    def get_num_image_tokens(
        self,
        image_size,
        max_slice_nums: int | None = None,
        downsample_mode: str | None = None,
    ) -> int:
        grid, source_tokens, patch_tokens = self._compute_visual_tokens(
            image_size,
            max_slice_nums=max_slice_nums,
            downsample_mode=downsample_mode,
        )
        return source_tokens + grid[0] * grid[1] * patch_tokens

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_size, downsample_mode="4x")

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}


class MiniCPMV4_7MultiModalProcessor(MiniCPMVMultiModalProcessor):
    def _resolve_downsample_mode(
        self,
        mm_kwargs: Mapping[str, object],
    ) -> str:
        downsample_mode = mm_kwargs.get("downsample_mode")
        if downsample_mode is not None:
            return str(downsample_mode)
        return self.info._get_downsample_mode()

    def _resolve_max_slice_nums(
        self,
        mm_kwargs: Mapping[str, object],
    ) -> int | None:
        max_slice_nums = mm_kwargs.get("max_slice_nums")
        return None if max_slice_nums is None else int(max_slice_nums)

    def get_image_prompt_texts(
        self,
        image_size,
        image_idx: int = 0,
        downsample_mode: str | None = None,
        max_slice_nums: int | None = None,
    ) -> str:
        return self.info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
            downsample_mode=downsample_mode,
            max_slice_nums=max_slice_nums,
        )

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (images := mm_data.get("images")) is None:
            return {}

        mm_items = self.info.parse_mm_data({"image": images}, validate=False)
        parsed_images = mm_items.get_items(
            "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
        )
        if isinstance(parsed_images, MiniCPMVImageEmbeddingItems):
            image_inputs = {}
        else:
            image_processor = self.info.get_image_processor()
            processor_kwargs = dict(mm_kwargs)
            processor_kwargs.setdefault("return_tensors", "pt")
            image_inputs = {
                "pixel_values": [],
                "image_sizes": [],
                "tgt_sizes": [],
            }
            for image in parsed_images:
                output = image_processor([image], **processor_kwargs)
                for key in image_inputs:
                    image_inputs[key].append(output[key][0])

        downsample_mode = self._resolve_downsample_mode(mm_kwargs)
        insert_layer_id = getattr(self.info.get_hf_config(), "insert_layer_id", 6)
        use_vit_merger = downsample_mode != "4x" and insert_layer_id >= 0
        if image_inputs:
            image_inputs["use_vit_merger"] = [
                torch.tensor([use_vit_merger], dtype=torch.bool)
                for _ in range(len(parsed_images))
            ]
        return image_inputs

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        downsample_mode = self._resolve_downsample_mode(hf_processor_mm_kwargs)
        max_slice_nums = self._resolve_max_slice_nums(hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        targets = [_VISION_PLACEHOLDER, _LEGACY_IMAGE_PLACEHOLDER]
        for target in tuple(targets):
            decoded = tokenizer.decode(
                tokenizer.encode(target, add_special_tokens=False)
            )
            if decoded != target:
                targets.append(decoded)

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
            )
            unk_token_id = tokenizer.unk_token_id
            if unk_token_id is None:
                raise ValueError("MiniCPM-V 4.7 tokenizer has no unk_token_id")
            placeholder = self.get_image_prompt_texts(
                images.get_image_size(item_idx),
                image_idx=item_idx,
                downsample_mode=downsample_mode,
                max_slice_nums=max_slice_nums,
            )
            return PromptUpdateDetails.select_token_id(
                tokenizer.encode(placeholder, add_special_tokens=False),
                unk_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=target,
                replacement=get_replacement,
            )
            for target in dict.fromkeys(targets)
        ]

    def _recompute_cached_prompt_update(
        self,
        cached_update: ResolvedPromptUpdate,
        new_item_idx: int,
    ) -> ResolvedPromptUpdate:
        new_update = super()._recompute_cached_prompt_update(
            cached_update,
            new_item_idx,
        )
        if cached_update.modality != "image":
            return new_update

        tokenizer = self.info.get_tokenizer()
        unk_token_id = tokenizer.unk_token_id
        if unk_token_id is None:
            raise ValueError("MiniCPM-V 4.7 tokenizer has no unk_token_id")
        content = new_update.content.full
        if isinstance(content, str):
            content = tokenizer.encode(content, add_special_tokens=False)
        return new_update.with_content(
            PromptUpdateDetails.select_token_id(content, unk_token_id)
        )

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmv4_7_field_config(hf_inputs)


class ViTMergerAttention4_7(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_windows, window_size, hidden_size = x.shape
        q = self.q_proj(x).view(num_windows, window_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_windows, window_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_windows, window_size, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )
        output = output.transpose(1, 2).reshape(num_windows, window_size, hidden_size)
        return self.out_proj(output)


class ViTMergerMLP4_7(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        from vllm.model_executor.layers.activation import get_act_fn

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class ViTWindowAttentionMerger4_7(nn.Module):
    def __init__(self, vision_config: PretrainedConfig):
        super().__init__()
        hidden_size = vision_config.hidden_size
        intermediate_size = vision_config.intermediate_size
        hidden_act = getattr(vision_config, "hidden_act", "gelu_pytorch_tanh")

        self.window_kernel_size = (2, 2)
        self.self_attn = ViTMergerAttention4_7(
            hidden_size,
            vision_config.num_attention_heads,
        )
        self.layer_norm1 = nn.LayerNorm(
            hidden_size,
            eps=vision_config.layer_norm_eps,
        )
        self.mlp = ViTMergerMLP4_7(
            hidden_size,
            intermediate_size,
            hidden_act,
        )
        self.layer_norm2 = nn.LayerNorm(
            hidden_size,
            eps=vision_config.layer_norm_eps,
        )

        merged_hidden_size = hidden_size * 4
        merged_intermediate_size = intermediate_size * 4
        self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(
            merged_hidden_size,
            merged_intermediate_size,
            bias=True,
        )
        from vllm.model_executor.layers.activation import get_act_fn

        self.act = get_act_fn(hidden_act)
        self.linear_2 = nn.Linear(
            merged_intermediate_size,
            hidden_size,
            bias=True,
        )

    def _apply_window_attention(
        self,
        valid_states: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        hidden_size = valid_states.shape[-1]
        num_height_windows = height // 2
        num_width_windows = width // 2
        x = valid_states.view(height, width, hidden_size)
        x = x.view(num_height_windows, 2, num_width_windows, 2, hidden_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(num_height_windows * num_width_windows, 4, hidden_size)
        x = self.self_attn(x)
        x = x.view(
            num_height_windows,
            num_width_windows,
            2,
            2,
            hidden_size,
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(height * width, hidden_size)

    def _apply_mlp_downsample(
        self,
        valid_states: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        hidden_size = valid_states.shape[-1]
        num_height_windows = height // 2
        num_width_windows = width // 2
        x = valid_states.view(height, width, hidden_size)
        x = x.view(num_height_windows, 2, num_width_windows, 2, hidden_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        residual = x.reshape(
            num_height_windows * num_width_windows, 4, hidden_size
        ).mean(dim=1)
        x = x.reshape(num_height_windows * num_width_windows, 4 * hidden_size)
        x = self.pre_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x + residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, _, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        residual = hidden_states.clone()
        normed = self.layer_norm1(hidden_states)

        attention_results = torch.zeros_like(hidden_states)
        for index in range(batch_size):
            height, width = tgt_sizes[index].tolist()
            valid = normed[index, : height * width, :]
            attention_results[index, : height * width, :] = (
                self._apply_window_attention(valid, height, width)
            )
        hidden_states = residual + attention_results

        new_tgt_sizes = torch.zeros_like(tgt_sizes)
        downsampled = []
        for index in range(batch_size):
            height, width = tgt_sizes[index].tolist()
            valid = hidden_states[index, : height * width, :]
            downsampled.append(self._apply_mlp_downsample(valid, height, width))
            new_tgt_sizes[index] = torch.tensor(
                [height // 2, width // 2],
                device=device,
                dtype=tgt_sizes.dtype,
            )

        new_num_patches = new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1]
        max_new_patches = int(new_num_patches.max().item())
        new_hidden_states = torch.zeros(
            (batch_size, max_new_patches, hidden_size),
            dtype=dtype,
            device=device,
        )
        for index, item in enumerate(downsampled):
            new_hidden_states[index, : item.shape[0], :] = item

        new_attention_mask = None
        if attention_mask is not None:
            mask = torch.zeros(
                (batch_size, max_new_patches),
                dtype=torch.bool,
                device=device,
            )
            for index in range(batch_size):
                mask[index, : int(new_num_patches[index].item())] = True
            min_value = torch.finfo(dtype).min
            new_attention_mask = (~mask).to(dtype=dtype) * min_value
            new_attention_mask = new_attention_mask[:, None, None, :]
        return new_hidden_states, new_tgt_sizes, new_attention_mask


class DownsampleMLP4_7(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.hidden_size = hidden_size * merge_kernel_size[0] * merge_kernel_size[1]
        self.pre_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pre_norm(x))


class Merger4_7(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: tuple[int, int] = (2, 2),
        times: int = 1,
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.times = times
        self.mlp = nn.ModuleList(
            [
                DownsampleMLP4_7(
                    hidden_size,
                    llm_embed_dim if index == times - 1 else hidden_size,
                    merge_kernel_size,
                )
                for index in range(times)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        kernel_height, kernel_width = self.merge_kernel_size
        processed = []
        for index in range(hidden_states.shape[0]):
            height, width = (int(value) for value in tgt_sizes[index].tolist())
            num_patches = height * width
            x = hidden_states[index, :num_patches, :]
            x = x.view(
                height // kernel_height,
                kernel_height,
                width // kernel_width,
                kernel_width,
                -1,
            )
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(
                height // kernel_height * width // kernel_width,
                -1,
            )
            x = self.mlp[0](x)

            current_height = height // kernel_height
            current_width = width // kernel_width
            for merger in self.mlp[1:]:
                x = x.view(current_height, current_width, -1)
                x = x.view(
                    current_height // kernel_height,
                    kernel_height,
                    current_width // kernel_width,
                    kernel_width,
                    -1,
                )
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                x = x.view(
                    current_height // kernel_height * current_width // kernel_width,
                    -1,
                )
                x = merger(x)
                current_height //= kernel_height
                current_width //= kernel_width
            processed.append(x)
        return processed


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_7MultiModalProcessor,
    info=MiniCPMV4_7ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_7ForConditionalGeneration(
    MiniCPMVBaseModel,
    SupportsLoRA,
    HasInnerState,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return _LEGACY_IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        version = tuple(int(part) for part in str(config.version).split("."))
        if version != (4, 7):
            raise ValueError(f"Unsupported MiniCPM-V OCR version: {config.version}")

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.insert_layer_id = getattr(config, "insert_layer_id", 6)
        self.vit_merger = ViTWindowAttentionMerger4_7(config.vision_config)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        config = vllm_config.model_config.hf_config
        saved_model_type = config.model_type
        config.model_type = "qwen3_5_text"
        try:
            return Qwen3_5ForCausalLM(
                vllm_config=vllm_config,
                prefix=prefix,
            )
        finally:
            config.model_type = saved_model_type

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        model.apply_encoder_attention_mask = True
        if config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        return Merger4_7(
            hidden_size=vision_dim,
            llm_embed_dim=embed_dim,
        )

    def get_vision_hidden_states(
        self,
        data: MiniCPMVImagePixelInputs,
        downsample_mode: str | None = None,
    ) -> list[torch.Tensor]:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]
        batch_size = len(pixel_values)
        patch_size = pixel_values[0].shape[-2]
        max_length = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros(
            (batch_size, 3, patch_size, max_length),
            dtype=dtype,
            device=device,
        )
        for index, item in enumerate(pixel_values):
            all_pixel_values[index, ..., : item.shape[-1]] = item

        num_patches = tgt_sizes.prod(-1)
        max_patches = int(num_patches.max().item())
        patch_attention_mask = torch.zeros(
            (batch_size, max_patches),
            dtype=torch.bool,
            device=device,
        )
        for index, item_num_patches in enumerate(num_patches):
            patch_attention_mask[index, :item_num_patches] = True

        hidden_states = self.vpm.embeddings(
            pixel_values=all_pixel_values,
            patch_attention_mask=patch_attention_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )
        if self.vpm.apply_encoder_attention_mask and torch.any(~patch_attention_mask):
            min_value = torch.finfo(hidden_states.dtype).min
            attention_mask = (~patch_attention_mask).to(
                dtype=hidden_states.dtype
            ) * min_value
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = None

        if downsample_mode is None:
            downsample_mode = getattr(
                self,
                "_runtime_downsample_mode",
                None,
            ) or getattr(self.config, "downsample_mode", "16x")
        use_vit_merger = downsample_mode != "4x" and self.insert_layer_id >= 0

        for layer in self.vpm.encoder.layers[: self.insert_layer_id + 1]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        if use_vit_merger:
            hidden_states, tgt_sizes, attention_mask = self.vit_merger(
                hidden_states,
                tgt_sizes,
                attention_mask,
            )
        for layer in self.vpm.encoder.layers[self.insert_layer_id + 1 :]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
        hidden_states = self.vpm.post_layernorm(hidden_states)
        return self.resampler(hidden_states, tgt_sizes)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        use_vit_merger = kwargs.pop("use_vit_merger", None)
        if use_vit_merger is None:
            self._runtime_downsample_mode = None
        elif isinstance(use_vit_merger, torch.Tensor):
            self._runtime_downsample_mode = (
                "16x" if bool(use_vit_merger.any().item()) else "4x"
            )
        elif isinstance(use_vit_merger, list | tuple):
            flag = any(
                bool(item.any().item())
                if isinstance(item, torch.Tensor)
                else bool(item)
                for item in use_vit_merger
            )
            self._runtime_downsample_mode = "16x" if flag else "4x"
        else:
            self._runtime_downsample_mode = "16x" if bool(use_vit_merger) else "4x"
        return super().embed_multimodal(**kwargs)

    def _process_vision_input(
        self,
        image_input: MiniCPMVImageInputs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        vision_outputs = self.get_vision_hidden_states(image_input)
        result = []
        start = 0
        for num_slices in image_input["num_slices"].tolist():
            group = vision_outputs[start : start + num_slices]
            result.append(torch.cat(group, dim=0))
            start += num_slices
        return result

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="llm",
            connector="resampler",
            tower_model=["vpm", "vit_merger"],
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
