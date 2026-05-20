# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import yaml
import torch
from loguru import logger
from osc_data.video import Video

from flash_talk.src.pipeline.flash_talk_pipeline import FlashTalkPipeline as LegacyFlashTalkPipeline
from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree

from flash_talk.infinite_talk.configs import multitalk_14B
from flash_talk.infinite_talk.utils.multitalk_utils import loudness_norm

with open(Path(__file__).parent / "configs" / "infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

# TODO: support more resolution
target_size = (infer_params['height'], infer_params['width'])

def get_pipeline(
    world_size,
    ckpt_dir,
    wav2vec_dir,
    cpu_offload=False,
    keep_dit_on_gpu=False,
    num_persistent_param_in_dit=15_000_000_000,
    t5_quant=None,
    t5_quant_dir=None,
    helper_cpu_offload=True,
):
    cfg = multitalk_14B

    ulysses_degree, ring_degree = get_parallel_degree(world_size, cfg.num_heads)
    device = get_device(ulysses_degree, ring_degree)
    logger.info(f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}")

    pipeline = LegacyFlashTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
        cpu_offload=cpu_offload,
        keep_dit_on_gpu=keep_dit_on_gpu,
        num_persistent_param_in_dit=num_persistent_param_in_dit,
        t5_quant=t5_quant,
        t5_quant_dir=t5_quant_dir,
    )
    if not helper_cpu_offload:
        logger.warning(
            "helper_cpu_offload is ignored in clean-copy; the legacy pipeline handles offload itself."
        )

    return pipeline

def get_base_data(pipeline, input_prompt, cond_image, base_seed, target_size=target_size):
    pipeline.prepare_params(
        input_prompt=input_prompt, 
        cond_image=cond_image,
        target_size=target_size,
        frame_num=infer_params['frame_num'],
        motion_frames_num=infer_params['motion_frames_num'],
        sampling_steps=infer_params['sample_steps'],
        seed=base_seed,
        shift=infer_params['sample_shift'],
        color_correction_strength=infer_params['color_correction_strength'],
    )

def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    audio_array = loudness_norm(audio_array, infer_params['sample_rate'])
    audio_embedding = pipeline.preprocess_audio(audio_array, sr=infer_params['sample_rate'], fps=infer_params['tgt_fps'])

    if audio_start_idx == -1 or audio_end_idx == -1:
        audio_start_idx = 0
        audio_end_idx = audio_embedding.shape[0]

    indices = (torch.arange(2 * 2 + 1) - 2) * 1

    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx-1)

    audio_embedding = audio_embedding[center_indices][None,...].contiguous()
    return audio_embedding

def run_pipeline(pipeline, audio_embedding):
    audio_embedding = audio_embedding.to(pipeline.device)
    sample = pipeline.generate(audio_embedding)
    sample_frames = (((sample+1)/2).permute(1,2,3,0).clip(0,1) * 255).contiguous()
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


class GeneratedVideoArtifact:
    """Lightweight file-backed generated video handle."""

    def __init__(self, uri: str, prompt: str, fps: int, frame_count: int, width: int, height: int, has_audio: bool = False):
        self.uri = uri
        self.prompt = prompt
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.has_audio = has_audio

    def save(self, path: str, format: str | None = None, codec: str | None = None):
        del format, codec
        if os.path.abspath(path) == os.path.abspath(self.uri):
            return self
        parent_dir = os.path.dirname(os.fspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        shutil.copy2(self.uri, path)
        return GeneratedVideoArtifact(path, self.prompt, self.fps, self.frame_count, self.width, self.height, has_audio=self.has_audio)

    def _resolve_audio_uri(self, audio) -> str:
        audio_uri = getattr(audio, "uri", None)
        if audio_uri:
            return audio_uri
        audio_path = getattr(audio, "path", None)
        if audio_path:
            return audio_path
        if isinstance(audio, (str, os.PathLike)):
            return os.fspath(audio)
        raise ValueError("Audio object must expose a uri or path for ffmpeg muxing.")

    def merge_audio(self, audio, output_path: str, audio_mode: str = "loop"):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for merge_audio but was not found on PATH.")

        parent_dir = os.path.dirname(os.fspath(output_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", self.uri]
        if audio_mode == "loop":
            cmd.extend(["-stream_loop", "-1", "-i", self._resolve_audio_uri(audio)])
        elif audio_mode == "silence":
            sample_rate = int(getattr(audio, "sampling_rate", getattr(audio, "sample_rate", 16000)))
            duration = max(self.frame_count / max(self.fps, 1), 0.0)
            cmd.extend([
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
                "-t",
                str(duration),
            ])
        else:
            raise ValueError("audio_mode must be 'loop' or 'silence'")

        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
        subprocess.run(cmd, check=True)
        return GeneratedVideoArtifact(output_path, self.prompt, self.fps, self.frame_count, self.width, self.height, has_audio=True)


class _FFmpegRawVideoWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for streaming output but was not found on PATH.")
        self._closed = False
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_batch: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed ffmpeg stream.")
        if frame_batch.dtype != np.uint8 or not frame_batch.flags.c_contiguous:
            frame_batch = np.ascontiguousarray(frame_batch, dtype=np.uint8)
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available.")
        self._proc.stdin.write(memoryview(frame_batch))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stderr_text = ""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._proc.stderr is not None:
            stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace")
            self._proc.stderr.close()
        return_code = self._proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg raw video writer failed with exit code {return_code}: {stderr_text.strip()}"
            )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FlashTalkInferencePipeline:
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
        self.pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            cpu_offload=cpu_offload,
            keep_dit_on_gpu=keep_dit_on_gpu,
            num_persistent_param_in_dit=num_persistent_param_in_dit,
            t5_quant=t5_quant,
            t5_quant_dir=t5_quant_dir,
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
    ) -> GeneratedVideoArtifact:
        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        cached_audio_duration = infer_params["cached_audio_duration"]
        frame_num = infer_params["frame_num"] if frame_num is None else frame_num
        motion_frames_num = infer_params["motion_frames_num"] if motion_frames_num is None else motion_frames_num
        sampling_steps = infer_params["sample_steps"] if sampling_steps is None else sampling_steps
        shift = infer_params["sample_shift"] if shift is None else shift
        color_correction_strength = infer_params["color_correction_strength"] if color_correction_strength is None else color_correction_strength
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
        chunk_count = 0

        output_dir = "sample_results"
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"nb_{os.getpid()}_{seed}.temp.mp4")
        writer = _FFmpegRawVideoWriter(temp_path, target_size[1], target_size[0], tgt_fps)
        logger.info(
            f"Notebook generation started: audio_encode_mode={audio_encode_mode}, target_size={target_size}, temp_path={temp_path}"
        )

        try:
            if audio_encode_mode == "once":
                human_speech_array_frame_num = frame_num * sample_rate // tgt_fps
                human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
                remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
                if remainder > 0:
                    pad_length = human_speech_array_slice_len - remainder
                    human_speech_array_all = np.concatenate(
                        [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
                    )
                else:
                    pad_length = 0

                audio_embedding_all = get_audio_embedding(self.pipeline, human_speech_array_all)
                audio_embedding_chunks_list = [
                    audio_embedding_all[:, i * slice_len : i * slice_len + frame_num].contiguous()
                    for i in range((audio_embedding_all.shape[1] - frame_num) // slice_len)
                ]
                logger.info(
                    f"Notebook once-mode: audio_len={len(human_speech_array_all)}, pad_len={pad_length}, chunk_count={len(audio_embedding_chunks_list)}"
                )
                for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    video = run_pipeline(self.pipeline, audio_embedding_chunk)
                    if chunk_idx != 0:
                        video = video[motion_frames_num:]
                    torch.cuda.synchronize()
                    end_time = time.time()
                    logger.info(f"Notebook chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")
                    writer.write(video.detach().cpu().numpy())
                    chunk_count += 1

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
                        [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
                    )
                else:
                    pad_length = 0
                human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)
                logger.info(
                    f"Notebook stream-mode: audio_len={len(human_speech_array_all)}, pad_len={pad_length}, chunk_count={len(human_speech_array_slices)}"
                )
                for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
                    audio_dq.extend(human_speech_array.tolist())
                    audio_window = np.array(audio_dq)
                    audio_embedding = get_audio_embedding(self.pipeline, audio_window, audio_start_idx, audio_end_idx)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    video = run_pipeline(self.pipeline, audio_embedding)
                    video = video[motion_frames_num:]
                    torch.cuda.synchronize()
                    end_time = time.time()
                    logger.info(f"Notebook chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")
                    writer.write(video.detach().cpu().numpy())
                    chunk_count += 1
            else:
                raise ValueError("audio_encode_mode must be 'once' or 'stream'")
        finally:
            writer.close()

        artifact = GeneratedVideoArtifact(
            temp_path,
            input_prompt,
            tgt_fps,
            frame_count=max(1, chunk_count * slice_len + motion_frames_num),
            width=target_size[1],
            height=target_size[0],
        )
        logger.info(f"Notebook generation complete: {temp_path}")
        return artifact


# Fast-copy compatible public alias.
FlashTalkPipeline = FlashTalkInferencePipeline

# Backward-compatible aliases for older notebook cells.
FlashTalkNotebookPipeline = FlashTalkInferencePipeline
NotebookFlashTalkPipeline = FlashTalkInferencePipeline
