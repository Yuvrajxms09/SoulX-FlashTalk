from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class _LoadedAudio:
    data: np.ndarray


@dataclass(frozen=True, slots=True)
class Audio:
    uri: str

    @property
    def path(self) -> str:
        return self.uri

    def __fspath__(self) -> str:
        return self.uri

    def load(self, sample_rate: int = 16000, mono: bool = True) -> _LoadedAudio:
        channel_args = ["-ac", "1"] if mono else []
        cmd = [
            "ffmpeg",
            "-i",
            self.uri,
            "-f",
            "f32le",
            *channel_args,
            "-ar",
            str(sample_rate),
            "-hide_banner",
            "-loglevel",
            "error",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        return _LoadedAudio(data=np.frombuffer(proc.stdout, dtype=np.float32))
