from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import imageio
import numpy as np


def _resolve_audio_path(audio: Any) -> str | None:
    if isinstance(audio, (str, os.PathLike)):
        return os.fspath(audio)
    for attr in ("uri", "path"):
        value = getattr(audio, attr, None)
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
    return None


@dataclass(slots=True)
class Video:
    data: np.ndarray
    prompt: str | None = None
    fps: int = 25

    def save(self, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with imageio.get_writer(
            output_path,
            format="mp4",
            mode="I",
            fps=self.fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frame in self.data:
                writer.append_data(frame)

    def merge_audio(self, audio: Any, output_path: str) -> None:
        audio_path = _resolve_audio_path(audio)
        if audio_path is None:
            raise ValueError("audio must be a path or expose uri/path")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video_path = tmp.name

        try:
            self.save(temp_video_path)
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
