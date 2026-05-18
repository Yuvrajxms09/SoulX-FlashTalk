from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage


@dataclass(frozen=True, slots=True)
class _LoadedImage:
    data: np.ndarray

    def to_rgb(self) -> "_LoadedImage":
        return self


@dataclass(frozen=True, slots=True)
class Image:
    uri: str

    @property
    def path(self) -> str:
        return self.uri

    def __fspath__(self) -> str:
        return self.uri

    def load(self) -> _LoadedImage:
        image = PILImage.open(self.uri).convert("RGB")
        return _LoadedImage(data=np.array(image, dtype=np.uint8))
