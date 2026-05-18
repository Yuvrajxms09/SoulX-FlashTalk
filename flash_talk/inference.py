"""High-level inference helpers for FlashTalk."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger

from flash_talk.infinite_talk.configs import multitalk_14B
from flash_talk.infinite_talk.utils.multitalk_utils import loudness_norm
from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree
from flash_talk.src.pipeline.flash_talk_pipeline import FlashTalkPipeline as RawFlashTalkPipeline

with open(Path(__file__).parent / "configs" / "infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

# TODO: support more resolution
target_size = (infer_params["height"], infer_params["width"])


@dataclass
class NotebookVideo:
    frames: torch.Tensor
    fps: int

    def save(self, output_path: str) -> None:
        _write_video_frames(self.frames, output_path, self.fps)

    def merge_audio(self, audio: Any, output_path: str) -> None:
        audio_path = _resolve_audio_path(audio)
        if audio_path is None:
            raise ValueError("merge_audio requires an audio file path or an object with uri/path")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video_path = tmp.name

        try:
            _write_video_frames(self.frames, temp_video_path, self.fps)
            cmd = [
                "ffmpeg",
                "-i",
                temp_video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
                "-y",
            ]
            subprocess.run(cmd, check=True)
        finally:
            try:
                os.remove(temp_video_path)
            except OSError:
                pass


def get_pipeline(world_size, ckpt_dir, wav2vec_dir, cpu_offload=False, helper_cpu_offload=True):
    cfg = multitalk_14B

    ulysses_degree, ring_degree = get_parallel_degree(world_size, cfg.num_heads)
    device = get_device(ulysses_degree, ring_degree)
    logger.info(
        f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}"
    )

    pipeline = RawFlashTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
        cpu_offload=cpu_offload,
        helper_cpu_offload=helper_cpu_offload,
    )

    return pipeline


def get_base_data(pipeline, input_prompt, cond_image, base_seed):
    pipeline.prepare_params(
        input_prompt=input_prompt,
        cond_image=cond_image,
        target_size=target_size,
        frame_num=infer_params["frame_num"],
        motion_frames_num=infer_params["motion_frames_num"],
        sampling_steps=infer_params["sample_steps"],
        seed=base_seed,
        shift=infer_params["sample_shift"],
        color_correction_strength=infer_params["color_correction_strength"],
    )


def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    audio_array = loudness_norm(audio_array, infer_params["sample_rate"])
    audio_embedding = pipeline.preprocess_audio(
        audio_array,
        sr=infer_params["sample_rate"],
        fps=infer_params["tgt_fps"],
    )

    if audio_start_idx == -1 or audio_end_idx == -1:
        audio_start_idx = 0
        audio_end_idx = audio_embedding.shape[0]

    indices = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)

    audio_embedding = audio_embedding[center_indices][None, ...].contiguous()
    return audio_embedding


def run_pipeline(pipeline, audio_embedding):
    audio_embedding = audio_embedding.to(pipeline.device)
    sample = pipeline.generate(audio_embedding)
    sample_frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous()
    return sample_frames


def _resolve_image_path(image: Any) -> str | None:
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    for attr in ("uri", "path"):
        value = getattr(image, attr, None)
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
    return None


def _resolve_audio_path(audio: Any) -> str | None:
    if isinstance(audio, (str, os.PathLike)):
        return os.fspath(audio)
    for attr in ("uri", "path"):
        value = getattr(audio, attr, None)
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
    return None


def _load_audio_array(audio: Any, sample_rate: int) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio
    audio_path = _resolve_audio_path(audio)
    if audio_path is None:
        raise TypeError("audio must be a numpy array, file path, or object with uri/path")

    cmd = [
        "ffmpeg",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-hide_banner",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _write_video_frames(frames: torch.Tensor, output_path: str, fps: int) -> None:
    import imageio

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with imageio.get_writer(
        output_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        video = frames.detach().cpu().numpy().astype(np.uint8)
        for idx in range(video.shape[0]):
            writer.append_data(video[idx])


class NotebookFlashTalkPipeline:
    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        *,
        world_size: int = 1,
        cpu_offload: bool = False,
        helper_cpu_offload: bool = True,
        base_seed: int = 9999,
    ) -> None:
        self.pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            cpu_offload=cpu_offload,
            helper_cpu_offload=helper_cpu_offload,
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
    ) -> NotebookVideo:
        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        cached_audio_duration = infer_params["cached_audio_duration"]
        frame_num = infer_params["frame_num"] if frame_num is None else frame_num
        motion_frames_num = (
            infer_params["motion_frames_num"]
            if motion_frames_num is None
            else motion_frames_num
        )
        sampling_steps = infer_params["sample_steps"] if sampling_steps is None else sampling_steps
        shift = infer_params["sample_shift"] if shift is None else shift
        color_correction_strength = (
            infer_params["color_correction_strength"]
            if color_correction_strength is None
            else color_correction_strength
        )
        target_size = target_size or (infer_params["height"], infer_params["width"])
        seed = self.base_seed if seed is None else seed
        image = _resolve_image_path(image) or image

        self.pipeline.prepare_params(
            input_prompt=input_prompt,
            cond_image=image,
            target_size=target_size,
            frame_num=frame_num,
            motion_frames_num=motion_frames_num,
            sampling_steps=sampling_steps,
            seed=seed,
            shift=shift,
            color_correction_strength=color_correction_strength,
        )

        audio_array = _load_audio_array(audio, sample_rate)
        human_speech_array_all = loudness_norm(audio_array, sample_rate)
        slice_len = frame_num - motion_frames_num

        generated_list: list[torch.Tensor] = []
        if audio_encode_mode == "once":
            human_speech_array_frame_num = frame_num * sample_rate // tgt_fps
            human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
            remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                human_speech_array_all = np.concatenate(
                    [
                        human_speech_array_all,
                        np.zeros(pad_length, dtype=human_speech_array_all.dtype),
                    ]
                )

            audio_embedding_all = get_audio_embedding(self.pipeline, human_speech_array_all)
            audio_embedding_len = audio_embedding_all.shape[1]
            chunk_count = max(1, (audio_embedding_len - frame_num + slice_len) // slice_len)
            audio_embedding_chunks_list = [
                audio_embedding_all[:, i * slice_len : i * slice_len + frame_num].contiguous()
                for i in range(chunk_count)
            ]

            for audio_embedding_chunk in audio_embedding_chunks_list:
                video = run_pipeline(self.pipeline, audio_embedding_chunk)
                generated_list.append(video.cpu())
        elif audio_encode_mode == "stream":
            cached_audio_length_sum = sample_rate * cached_audio_duration
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num

            audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
            human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

            remainder = len(human_speech_array_all) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                human_speech_array_all = np.concatenate(
                    [
                        human_speech_array_all,
                        np.zeros(pad_length, dtype=human_speech_array_all.dtype),
                    ]
                )

            human_speech_array_slices = human_speech_array_all.reshape(
                -1, human_speech_array_slice_len
            )
            for human_speech_array in human_speech_array_slices:
                audio_dq.extend(human_speech_array.tolist())
                audio_array_window = np.array(audio_dq)
                audio_embedding = get_audio_embedding(
                    self.pipeline, audio_array_window, audio_start_idx, audio_end_idx
                )
                video = run_pipeline(self.pipeline, audio_embedding)
                generated_list.append(video.cpu())
        else:
            raise ValueError("audio_encode_mode must be 'stream' or 'once'")

        frames = (
            torch.cat(generated_list, dim=0) if len(generated_list) > 1 else generated_list[0]
        )
        return NotebookVideo(frames=frames, fps=tgt_fps)


FlashTalkPipeline = NotebookFlashTalkPipeline
