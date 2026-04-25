import os
import tempfile
from dataclasses import replace
from typing import Any

import torch

from opensora.datasets import save_sample
from opensora.datasets.utils import read_from_path
from opensora.utils.sampling import SamplingOption, sanitize_sampling_option


class LongFormGenerationError(RuntimeError):
    """Raised when long-form generation receives invalid settings."""


def _blend_overlap(prev_chunk: torch.Tensor, new_chunk: torch.Tensor, overlap: int) -> torch.Tensor:
    """Blend the temporal overlap between chunks using linear cross-fade."""
    if overlap <= 0:
        return torch.cat([prev_chunk, new_chunk], dim=2)

    overlap = min(overlap, prev_chunk.shape[2], new_chunk.shape[2])
    if overlap == 0:
        return torch.cat([prev_chunk, new_chunk], dim=2)

    alpha = torch.linspace(0, 1, overlap, device=prev_chunk.device, dtype=prev_chunk.dtype).view(1, 1, overlap, 1, 1)
    blended = prev_chunk[:, :, -overlap:] * (1 - alpha) + new_chunk[:, :, :overlap] * alpha
    return torch.cat([prev_chunk[:, :, :-overlap], blended, new_chunk[:, :, overlap:]], dim=2)


def _save_temp_video(video: torch.Tensor, fps: int, prefix: str) -> str:
    """Persist a single video tensor [C, T, H, W] to a temporary mp4 path."""
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    base = os.path.join(tmp_dir, "clip")
    save_sample(video.detach().cpu(), save_path=base, fps=fps, force_video=True, verbose=False)
    return f"{base}.mp4"


def _chunk_lengths(total_frames: int, chunk_frames: int, overlap_frames: int) -> list[int]:
    if total_frames <= 0:
        raise LongFormGenerationError("total_frames must be > 0")
    if chunk_frames <= 0:
        raise LongFormGenerationError("chunk_frames must be > 0")
    if overlap_frames >= chunk_frames:
        raise LongFormGenerationError("overlap_frames must be smaller than chunk_frames")

    if total_frames <= chunk_frames:
        return [total_frames]

    lengths = [chunk_frames]
    generated = chunk_frames
    stride = chunk_frames - overlap_frames
    while generated < total_frames:
        needed = total_frames - generated
        step_len = min(chunk_frames, needed + overlap_frames)
        lengths.append(step_len)
        generated += stride
    return lengths


def generate_long_video(
    api_fn,
    sampling_option: SamplingOption,
    prompt: str,
    total_frames: int,
    chunk_frames: int = 97,
    overlap_frames: int = 16,
    fps: int = 24,
    adaptive_steps: bool = True,
    extension_prompt: str | None = None,
    seed: int | None = None,
    **api_kwargs: Any,
) -> torch.Tensor:
    """
    Autoregressive/chunked long-form generation.

    The first chunk uses plain t2v. Subsequent chunks use v2v_head with the last
    context from the stitched output as strong conditioning.
    """
    opt = sanitize_sampling_option(replace(sampling_option, num_frames=chunk_frames))
    chunk_plan = _chunk_lengths(total_frames=total_frames, chunk_frames=chunk_frames, overlap_frames=overlap_frames)

    stitched = None
    running_seed = seed

    for idx, local_frames in enumerate(chunk_plan):
        if adaptive_steps:
            step_scale = 1.0 + 0.05 * idx
            local_steps = max(16, int(round(sampling_option.num_steps * step_scale)))
        else:
            local_steps = sampling_option.num_steps

        local_opt = replace(opt, num_frames=local_frames, num_steps=local_steps)

        if idx == 0:
            chunk = api_fn(local_opt, cond_type="t2v", text=[prompt], seed=running_seed, **api_kwargs)
            stitched = chunk
        else:
            context_video = stitched[:, :, -max(32, overlap_frames):]
            context_ref_path = _save_temp_video(context_video[0], fps=fps, prefix="opensora_extend_context_")
            text = extension_prompt or prompt
            chunk = api_fn(
                local_opt,
                cond_type="v2v_head",
                text=[text],
                ref=[context_ref_path],
                seed=running_seed,
                **api_kwargs,
            )
            stitched = _blend_overlap(stitched, chunk, overlap=overlap_frames)

        if running_seed is not None:
            running_seed += 1

    return stitched[:, :, :total_frames]


def extend_video(
    api_fn,
    sampling_option: SamplingOption,
    source_video_path: str,
    continue_prompt: str,
    extend_frames: int,
    chunk_frames: int = 97,
    overlap_frames: int = 16,
    fps: int = 24,
    seed: int | None = None,
    **api_kwargs: Any,
) -> torch.Tensor:
    """Continue an existing clip with v2v-head conditioning and chunk stitching."""
    source = read_from_path(source_video_path, image_size=(sampling_option.height, sampling_option.width), transform_name="resize_crop")
    source = source.unsqueeze(0)

    generated = generate_long_video(
        api_fn=api_fn,
        sampling_option=sampling_option,
        prompt=continue_prompt,
        total_frames=extend_frames,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        fps=fps,
        extension_prompt=continue_prompt,
        seed=seed,
        **api_kwargs,
    )
    return torch.cat([source, generated], dim=2)


def remaster_video(
    api_fn,
    sampling_option: SamplingOption,
    source_video_path: str,
    remaster_prompt: str,
    fps: int = 24,
    denoise_strength: float = 0.35,
    **api_kwargs: Any,
) -> torch.Tensor:
    """
    Lightweight remaster pass.

    Uses v2v_head conditioning to preserve motion/layout and runs extra denoising steps
    to enhance detail, then blends with source to reduce temporal flicker.
    """
    if not (0.0 <= denoise_strength <= 1.0):
        raise LongFormGenerationError("denoise_strength must be in [0, 1]")

    source = read_from_path(source_video_path, image_size=(sampling_option.height, sampling_option.width), transform_name="resize_crop")
    source = source.unsqueeze(0)

    source_ref_path = _save_temp_video(source[0], fps=fps, prefix="opensora_remaster_ref_")
    remaster_opt = replace(
        sampling_option,
        num_frames=source.shape[2],
        num_steps=max(24, int(round(sampling_option.num_steps * (1.2 + denoise_strength)))),
        guidance=max(sampling_option.guidance, 5.0),
        guidance_img=max(sampling_option.guidance_img or 3.0, 4.0),
    )
    remaster_opt = sanitize_sampling_option(remaster_opt)

    refined = api_fn(
        remaster_opt,
        cond_type="v2v_head",
        text=[remaster_prompt],
        ref=[source_ref_path],
        **api_kwargs,
    )
    return source * (1.0 - denoise_strength) + refined * denoise_strength
