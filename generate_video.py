# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os

from loguru import logger

from flash_talk.inference import FlashTalkPipeline, infer_params


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify FlashTalk model checkpoint directory."
    assert args.wav2vec_dir is not None, "Please specify the wav2vec checkpoint directory."
    args.base_seed = args.base_seed if args.base_seed >= 0 else 9999


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate video from a text prompt or image using Wan")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to FlashTalk model checkpoint directory.")
    parser.add_argument("--wav2vec_dir", type=str, default=None, help="The path to the wav2vec checkpoint directory.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
    parser.add_argument("--base_seed", type=int, default=9999, help="The seed to use for generating the video.")
    parser.add_argument(
        "--input_prompt",
        type=str,
        default="A person is talking. Only the foreground characters are moving, the background remains static.",
        help="The prompt to generate the video.",
    )
    parser.add_argument("--height", type=int, default=infer_params["height"], help="Output video height.")
    parser.add_argument("--width", type=int, default=infer_params["width"], help="Output video width.")
    parser.add_argument(
        "--cond_image",
        type=str,
        default="examples/man.png",
        help="[meta file] The condition image path to generate the video.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="examples/cantonese_16k.wav",
        help="[meta file] The audio path to generate the video.",
    )
    parser.add_argument(
        "--audio_encode_mode",
        type=str,
        default="stream",
        choices=["stream", "once"],
        help="stream: encode audio chunk before every generation; once: encode audio together",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload for low VRAM usage",
    )
    parser.add_argument(
        "--keep_dit_on_gpu",
        action="store_true",
        help="Keep the DiT model resident on GPU instead of using VRAM management.",
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=15_000_000_000,
        help="Target persistent parameter budget for VRAM management.",
    )
    parser.add_argument(
        "--t5_quant",
        type=str,
        default=None,
        choices=["int8", "fp8"],
        help="Optional quantized T5 loading mode.",
    )
    parser.add_argument(
        "--t5_quant_dir",
        type=str,
        default=None,
        help="Directory containing t5_int8.safetensors / t5_fp8.safetensors and map files.",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def generate(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    target_size = (args.height, args.width)
    logger.info(
        f"Starting generation: audio_encode_mode={args.audio_encode_mode}, target_size={target_size}, save_file={args.save_file}"
    )
    if not args.cpu_offload:
        logger.warning(
            "cpu_offload is disabled; this differs from the fast-copy default and will keep T5/CLIP/VAE resident longer."
        )

    notebook_pipeline = FlashTalkPipeline(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        world_size=world_size,
        cpu_offload=args.cpu_offload,
        keep_dit_on_gpu=args.keep_dit_on_gpu,
        num_persistent_param_in_dit=args.num_persistent_param_in_dit,
        t5_quant=args.t5_quant,
        t5_quant_dir=args.t5_quant_dir,
        base_seed=args.base_seed,
    )

    if args.save_file is None:
        output_dir = "sample_results"
        os.makedirs(output_dir, exist_ok=True)
        args.save_file = os.path.join(output_dir, f"res_{os.getpid()}.mp4")

    artifact = notebook_pipeline.generate(
        input_prompt=args.input_prompt,
        audio=args.audio_path,
        image=args.cond_image,
        audio_encode_mode=args.audio_encode_mode,
        target_size=target_size,
        frame_num=infer_params["frame_num"],
        motion_frames_num=infer_params["motion_frames_num"],
        sampling_steps=infer_params["sample_steps"],
        shift=infer_params["sample_shift"],
        color_correction_strength=infer_params["color_correction_strength"],
    )
    temp_uri = artifact.uri
    artifact = artifact.merge_audio(args.audio_path, output_path=args.save_file)
    if temp_uri != args.save_file and os.path.exists(temp_uri):
        os.remove(temp_uri)
    logger.info(f"Saving generated video to {args.save_file}")
    logger.info("Finished.")
    return artifact


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
