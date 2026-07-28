# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vllm.model_executor.models.minicpmv import (
    MiniCPMVBaseModel,
    MiniCPMVMultiModalProcessor,
)
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)
from vllm.model_executor.models.minicpmv4_7 import (
    DownsampleMLP4_7,
    MiniCPMV4_7ForConditionalGeneration,
    MiniCPMV4_7MultiModalProcessor,
    MiniCPMV4_7ProcessingInfo,
    ViTWindowAttentionMerger4_7,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.multimodal.processing.processor import (
    PromptUpdateDetails,
    ResolvedPromptUpdate,
    UpdateMode,
)


def test_minicpmv4_7_accepts_chat_template_placeholder():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda value, **_: value
    tokenizer.decode.side_effect = lambda value: value
    processor = SimpleNamespace(
        _resolve_downsample_mode=lambda _: "4x",
        _resolve_max_slice_nums=lambda _: 9,
        info=SimpleNamespace(get_tokenizer=lambda: tokenizer),
    )

    updates = MiniCPMV4_7MultiModalProcessor._get_prompt_updates(
        processor,
        mm_items=MagicMock(),
        hf_processor_mm_kwargs={},
        out_mm_kwargs=MagicMock(),
    )

    assert any(
        update.target == "<|vision_start|><|image_pad|><|vision_end|>"
        for update in updates
    )


def test_minicpmv4_7_cached_update_selects_unk_tokens():
    tokenizer = MagicMock(
        unk_token_id=248077,
        image_token="<|image_pad|>",
    )
    tokenizer.encode.return_value = [248090, 0, 248091, 248078, 248077, 248079]
    processor = object.__new__(MiniCPMV4_7MultiModalProcessor)
    processor.info = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    cached_update = ResolvedPromptUpdate(
        modality="image",
        item_idx=0,
        mode=UpdateMode.REPLACE,
        target="(<image>./</image>)",
        content=PromptUpdateDetails.from_seq([248077]),
    )
    recomputed_update = ResolvedPromptUpdate(
        modality="image",
        item_idx=1,
        mode=UpdateMode.REPLACE,
        target="(<image>./</image>)",
        content=PromptUpdateDetails.select_text(
            "<image_id>1</image_id><image><unk></image>",
            tokenizer.image_token,
        ),
    )

    with patch.object(
        MiniCPMVMultiModalProcessor,
        "_recompute_cached_prompt_update",
        return_value=recomputed_update,
    ):
        result = MiniCPMV4_7MultiModalProcessor._recompute_cached_prompt_update(
            processor,
            cached_update,
            new_item_idx=1,
        )

    assert result.content.full == [248090, 0, 248091, 248078, 248077, 248079]
    assert result.content.is_embed is not None
    assert torch.equal(
        result.content.is_embed(tokenizer, result.content.full),
        torch.tensor([False, False, False, False, True, False]),
    )


def test_minicpmv4_7_uses_independent_model_path():
    assert issubclass(MiniCPMV4_7ForConditionalGeneration, MiniCPMVBaseModel)
    assert not issubclass(
        MiniCPMV4_7ForConditionalGeneration,
        MiniCPMV4_6ForConditionalGeneration,
    )

    mapping = MiniCPMV4_7ForConditionalGeneration.get_mm_mapping(None)
    assert mapping.language_model == ["llm"]
    assert mapping.connector == ["resampler"]
    assert mapping.tower_model == ["vpm", "vit_merger"]
    assert "hf_to_vllm_mapper" not in MiniCPMV4_7ForConditionalGeneration.__dict__


def test_minicpmv4_7_keeps_checkpoint_merger_parameters():
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
    )
    merger = ViTWindowAttentionMerger4_7(config)
    parameter_names = dict(merger.named_parameters())

    assert "layer_norm2.weight" in parameter_names
    assert "mlp.fc1.weight" in parameter_names
    assert "mlp.fc2.weight" in parameter_names

    downsample = DownsampleMLP4_7(hidden_size=8, llm_embed_dim=16)
    assert isinstance(downsample.mlp[0], nn.Linear)
    assert isinstance(downsample.mlp[1], nn.GELU)
    assert isinstance(downsample.mlp[2], nn.Linear)


def test_minicpmv4_7_routes_legacy_architecture():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(version=4.7),
    )
    assert (
        ModelRegistry._normalize_arch("MiniCPMV", model_config)
        == "MiniCPMV4_7ForConditionalGeneration"
    )

    unsupported_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(version=4.5),
    )
    assert (
        ModelRegistry._normalize_arch("MiniCPMV", unsupported_model_config)
        == "MiniCPMV"
    )


def test_minicpmv4_7_processing_version_and_slice_config():
    info = MagicMock(spec=MiniCPMV4_7ProcessingInfo)
    info.get_hf_config.return_value = SimpleNamespace(
        slice_config=SimpleNamespace(max_slice_nums=9)
    )

    assert MiniCPMV4_7ProcessingInfo.get_model_version(info) == (4, 7)
    assert MiniCPMV4_7ProcessingInfo.get_image_max_slice_num(info) == 9


def test_minicpmv4_7_profiles_4x_image_tokens():
    info = MagicMock(spec=MiniCPMV4_7ProcessingInfo)
    info.get_image_size_with_most_features.return_value = (1120, 10080)
    info.get_num_image_tokens.return_value = 2592

    assert MiniCPMV4_7ProcessingInfo.get_max_image_tokens(info) == 2592
    info.get_num_image_tokens.assert_called_once_with(
        (1120, 10080), downsample_mode="4x"
    )
    info.get_max_image_tokens.return_value = 2592
    assert MiniCPMV4_7ProcessingInfo.get_mm_max_tokens_per_item(
        info, seq_len=40960, mm_counts={"image": 1}
    ) == {"image": 2592}
