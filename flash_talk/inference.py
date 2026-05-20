# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

import yaml

from src.fast_flashtalk import Audio, FlashTalkPipeline as FastFlashTalkPipeline, Image
from src.fast_flashtalk.configs import multitalk_14B

with open(Path(__file__).parent / "configs" / "infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

__all__ = ["FlashTalkPipeline", "infer_params"]


def _resolve_path(value: Any) -> str | None:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    for attr in ("uri", "path"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (str, os.PathLike)):
            return os.fspath(candidate)
    return None


class FlashTalkPipeline:
    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        *,
        world_size: int = 1,
        cpu_offload: bool = True,
        keep_dit_on_gpu: bool = False,
        num_persistent_param_in_dit: int = 15_000_000_000,
        t5_quant: str | None = None,
        t5_quant_dir: str | None = None,
        base_seed: int = 9999,
    ) -> None:
        del world_size
        if not cpu_offload:
            warnings.warn(
                "cpu_offload is ignored here; the fast-copy pipeline always keeps this path enabled.",
                stacklevel=2,
            )
        if t5_quant_dir not in (None, ckpt_dir):
            warnings.warn(
                "t5_quant_dir is ignored here; fast-copy loads T5 quant files from the checkpoint directory.",
                stacklevel=2,
            )

        multitalk_14B.t5_quant = t5_quant
        self.pipeline = FastFlashTalkPipeline(
            checkpoint_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            keep_dit_on_gpu=keep_dit_on_gpu,
            num_persistent_param_in_dit=num_persistent_param_in_dit,
        )
        self.base_seed = base_seed

    @property
    def device(self):
        return self.pipeline.device

    def generate(
        self,
        input_prompt: str,
        audio: Any,
        image: Any,
        audio_encode_mode: str = "once",
        target_size: tuple[int, int] | None = None,
        frame_num: int | None = None,
        motion_frames_num: int | None = None,
        sampling_steps: int | None = None,
        seed: int | None = None,
        shift: int | None = None,
        color_correction_strength: float | None = None,
    ):
        # Keep the notebook-compatible arguments, but let the exact fast-copy pipeline
        # own the actual chunking, attention, and disk streaming path.
        del frame_num, motion_frames_num, sampling_steps, seed, shift, color_correction_strength

        audio_path = _resolve_path(audio)
        image_path = _resolve_path(image)
        if audio_path is None:
            raise TypeError("audio must be a file path or an object with uri/path")
        if image_path is None:
            raise TypeError("image must be a file path or an object with uri/path")

        audio_obj = audio if isinstance(audio, Audio) else Audio(uri=audio_path)
        image_obj = image if isinstance(image, Image) else Image(uri=image_path)

        return self.pipeline.generate(
            input_prompt=input_prompt,
            audio=audio_obj,
            image=image_obj,
            audio_encode_mode=audio_encode_mode,
            target_size=target_size,
        )
