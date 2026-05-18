from dataclasses import dataclass

from .inference import (
    FlashTalkPipeline,
    RawFlashTalkPipeline,
    NotebookFlashTalkPipeline,
    get_audio_embedding,
    get_base_data,
    get_pipeline,
    infer_params,
    run_pipeline,
)

__all__ = [
    "FlashTalkPipeline",
    "RawFlashTalkPipeline",
    "NotebookFlashTalkPipeline",
    "get_audio_embedding",
    "get_base_data",
    "get_pipeline",
    "infer_params",
    "run_pipeline",
]

@dataclass(frozen=True, slots=True)
class Image:
    uri: str

    @property
    def path(self) -> str:
        return self.uri

    def __fspath__(self) -> str:
        return self.uri


@dataclass(frozen=True, slots=True)
class Audio:
    uri: str

    @property
    def path(self) -> str:
        return self.uri

    def __fspath__(self) -> str:
        return self.uri


__all__.extend(["Image", "Audio"])
