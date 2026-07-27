# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    MiniCPMVModelConfig,
)
from vllm.model_executor.models.minicpmv import MiniCPMVProcessingInfo
from vllm.model_executor.models.minicpmv4_6 import MiniCPMV4_6ProcessingInfo
from vllm.multimodal.video import (
    MiniCPMVVideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)


@pytest.mark.parametrize(
    "architecture",
    ["MiniCPMV", "MiniCPMV4_6ForConditionalGeneration"],
)
def test_minicpmv_video_sampling_defaults(architecture: str):
    model_config = SimpleNamespace(multimodal_config=MultiModalConfig())

    config_cls = MODELS_CONFIG_MAP[architecture]
    assert config_cls is MiniCPMVModelConfig
    assert MiniCPMVVideoBackend.DEFAULT_FPS == config_cls.DEFAULT_VIDEO_FPS
    assert MiniCPMVVideoBackend.MAX_FRAMES == config_cls.MAX_VIDEO_FRAMES

    config_cls.verify_and_update_model_config(model_config)

    assert model_config.multimodal_config.media_io_kwargs["video"] == {
        "video_backend": "minicpmv",
        "fps": 1.0,
        "num_frames": 60,
    }


def test_minicpmv_video_sampling_preserves_user_overrides():
    model_config = SimpleNamespace(
        multimodal_config=MultiModalConfig(
            media_io_kwargs={
                "video": {
                    "video_backend": "opencv",
                    "fps": 2.0,
                    "num_frames": 24,
                }
            }
        )
    )

    MiniCPMVModelConfig.verify_and_update_model_config(model_config)

    assert model_config.multimodal_config.media_io_kwargs["video"] == {
        "video_backend": "opencv",
        "fps": 2.0,
        "num_frames": 24,
    }


@pytest.mark.parametrize(
    ("duration", "total_frames", "original_fps", "expected_frames"),
    [(9.72, 243, 25.0, 10), (30.0, 900, 30.0, 30), (120.0, 3600, 30.0, 60)],
)
def test_minicpmv_video_sampling_indices(
    duration: float,
    total_frames: int,
    original_fps: float,
    expected_frames: int,
):
    source = VideoSourceMetadata(
        total_frames_num=total_frames,
        original_fps=original_fps,
        duration=duration,
    )
    target = VideoTargetMetadata(
        num_frames=MiniCPMVModelConfig.MAX_VIDEO_FRAMES,
        fps=MiniCPMVModelConfig.DEFAULT_VIDEO_FPS,
        max_duration=300.0,
    )

    indices = MiniCPMVVideoBackend.compute_frames_index_to_sample(source, target)

    one_fps_indices = np.arange(0, total_frames, original_fps).astype(int)
    if len(one_fps_indices) > MiniCPMVModelConfig.MAX_VIDEO_FRAMES:
        sampled_positions = np.linspace(
            0,
            len(one_fps_indices) - 1,
            MiniCPMVModelConfig.MAX_VIDEO_FRAMES,
            dtype=int,
        )
        one_fps_indices = one_fps_indices[sampled_positions]

    assert len(indices) == expected_frames
    assert indices == one_fps_indices.tolist()


def test_minicpmv_profile_caps_each_video_at_60_frames():
    info = MagicMock(spec=MiniCPMVProcessingInfo)
    info.get_max_image_tokens.return_value = 0
    info.get_max_video_frames.return_value = 100

    num_frames = MiniCPMVProcessingInfo.get_num_frames_with_most_features(
        info,
        seq_len=100,
        mm_counts={"video": 1},
    )

    assert num_frames == 60


def test_minicpmv4_6_disables_slicing_for_video_frames():
    info = MagicMock(spec=MiniCPMV4_6ProcessingInfo)
    info.get_hf_config.return_value = SimpleNamespace(
        slice_config=SimpleNamespace(max_slice_nums=9)
    )

    image_max_slices = MiniCPMV4_6ProcessingInfo.get_image_max_slice_num(info)
    video_max_slices = MiniCPMV4_6ProcessingInfo.get_video_max_slice_num(info)

    assert image_max_slices == 9
    assert video_max_slices == 1
