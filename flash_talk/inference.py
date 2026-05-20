# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

from pathlib import Path

import yaml

from src.fast_flashtalk import Audio, FlashTalkPipeline, Image

with open(Path(__file__).parent / "configs" / "infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

__all__ = ["FlashTalkPipeline", "Image", "Audio", "infer_params"]

