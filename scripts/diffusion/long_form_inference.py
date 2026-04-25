import os
from pprint import pformat

import torch

from opensora.datasets import save_sample
from opensora.utils.cai import init_inference_environment
from opensora.utils.config import parse_alias, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import SamplingOption, prepare_api, prepare_models, sanitize_sampling_option
from opensora.utils.long_form import extend_video, generate_long_video, remaster_video


@torch.inference_mode()
def main():
    cfg = parse_alias(parse_configs())
    init_inference_environment()
    logger = create_logger()
    logger.info("Long-form inference configuration:\n%s", pformat(cfg.to_dict()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    os.makedirs(cfg.save_dir, exist_ok=True)

    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg,
        device,
        dtype,
        offload_model=cfg.get("offload_model", False),
    )
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    sampling_option = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))

    mode = cfg.long_form.mode
    fps = cfg.get("fps_save", 24)
    seed = cfg.get("seed", None)

    if mode == "generate":
        video = generate_long_video(
            api_fn=api_fn,
            sampling_option=sampling_option,
            prompt=cfg.prompt,
            total_frames=cfg.long_form.total_frames,
            chunk_frames=cfg.long_form.chunk_frames,
            overlap_frames=cfg.long_form.overlap_frames,
            adaptive_steps=cfg.long_form.get("adaptive_steps", True),
            extension_prompt=cfg.long_form.get("extension_prompt", None),
            fps=fps,
            seed=seed,
            patch_size=cfg.get("patch_size", 2),
            channel=cfg.model.in_channels,
        )
    elif mode == "extend":
        video = extend_video(
            api_fn=api_fn,
            sampling_option=sampling_option,
            source_video_path=cfg.long_form.source_video,
            continue_prompt=cfg.prompt,
            extend_frames=cfg.long_form.extend_frames,
            chunk_frames=cfg.long_form.chunk_frames,
            overlap_frames=cfg.long_form.overlap_frames,
            fps=fps,
            seed=seed,
            patch_size=cfg.get("patch_size", 2),
            channel=cfg.model.in_channels,
        )
    elif mode == "remaster":
        video = remaster_video(
            api_fn=api_fn,
            sampling_option=sampling_option,
            source_video_path=cfg.long_form.source_video,
            remaster_prompt=cfg.prompt,
            denoise_strength=cfg.long_form.get("denoise_strength", 0.35),
            fps=fps,
            patch_size=cfg.get("patch_size", 2),
            channel=cfg.model.in_channels,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    output_path = os.path.join(cfg.save_dir, cfg.long_form.get("output_name", f"{mode}_result"))
    save_sample(video[0].cpu(), save_path=output_path, fps=fps, force_video=True)
    logger.info("Saved result to %s.mp4", output_path)


if __name__ == "__main__":
    main()
