# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import yaml
import torch
from loguru import logger

from flash_talk.src.pipeline.flash_talk_pipeline import FlashTalkPipeline
from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree

from flash_talk.infinite_talk.configs import multitalk_14B
from flash_talk.infinite_talk.utils.multitalk_utils import loudness_norm

with open("flash_talk/configs/infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

# Env overrides for prod/Colab (optional)
if os.environ.get("SAMPLE_STEPS"):
    infer_params["sample_steps"] = int(os.environ["SAMPLE_STEPS"])
if os.environ.get("HEIGHT"):
    infer_params["height"] = int(os.environ["HEIGHT"])
if os.environ.get("WIDTH"):
    infer_params["width"] = int(os.environ["WIDTH"])
if os.environ.get("TGT_FPS"):
    infer_params["tgt_fps"] = int(os.environ["TGT_FPS"])
if os.environ.get("SAMPLE_RATE"):
    infer_params["sample_rate"] = int(os.environ["SAMPLE_RATE"])

target_size = (infer_params["height"], infer_params["width"])

def apply_infer_param_overrides(
    sample_steps=None, height=None, width=None, tgt_fps=None, sample_rate=None
):
    """Apply CLI/script overrides to infer_params. Call before get_pipeline/get_base_data."""
    global target_size
    if sample_steps is not None:
        infer_params["sample_steps"] = sample_steps
    if height is not None:
        infer_params["height"] = height
    if width is not None:
        infer_params["width"] = width
    if tgt_fps is not None:
        infer_params["tgt_fps"] = tgt_fps
    if sample_rate is not None:
        infer_params["sample_rate"] = sample_rate
    target_size = (infer_params["height"], infer_params["width"])

def get_pipeline(world_size, ckpt_dir, wav2vec_dir, cpu_offload=False):
    cfg = multitalk_14B

    ulysses_degree, ring_degree = get_parallel_degree(world_size, cfg.num_heads)
    device = get_device(ulysses_degree, ring_degree)
    logger.info(f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}")

    pipeline = FlashTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
        cpu_offload=cpu_offload,
    )

    return pipeline

def get_base_data(pipeline, input_prompt, cond_image, base_seed):
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

