#!/usr/bin/env python
"""Gradio UI for Open-Sora long-form generate/extend/remaster workflows.

Run:
    python gradio/long_form_app.py --config configs/diffusion/inference/long_form_30s.py --host 127.0.0.1 --port 7861
"""

import argparse
import os
import tempfile
from dataclasses import replace

import gradio as gr
import torch
from mmengine.config import Config

from opensora.datasets import save_sample
from opensora.utils.cai import init_inference_environment
from opensora.utils.config import parse_alias
from opensora.utils.long_form import extend_video, generate_long_video, remaster_video
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import SamplingOption, prepare_api, prepare_models, sanitize_sampling_option


def parse_args():
    parser = argparse.ArgumentParser(description="Open-Sora long-form Gradio app")
    parser.add_argument("--config", required=True, help="Path to base inference config")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="Host to bind")
    parser.add_argument("--port", default=7861, type=int, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


class Runtime:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        cfg = Config.fromfile(self.config_path)
        cfg = parse_alias(cfg)

        init_inference_environment()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

        model, model_ae, model_t5, model_clip, optional_models = prepare_models(
            cfg,
            self.device,
            self.dtype,
            offload_model=cfg.get("offload_model", False),
        )
        self.api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)
        self.base_sampling_option = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))
        self.in_channels = cfg.model.in_channels
        self.fps = cfg.get("fps_save", 24)
        self.seed = cfg.get("seed", None)
        self._loaded = True

    @torch.inference_mode()
    def run(
        self,
        mode: str,
        prompt: str,
        source_video: str | None,
        total_frames: int,
        extend_frames: int,
        chunk_frames: int,
        overlap_frames: int,
        denoise_strength: float,
        num_steps: int,
        guidance: float,
        guidance_img: float,
    ) -> str:
        self.load()

        local_opt = replace(
            self.base_sampling_option,
            num_steps=int(num_steps),
            guidance=float(guidance),
            guidance_img=float(guidance_img),
        )

        common_kwargs = dict(
            patch_size=self.cfg.get("patch_size", 2),
            channel=self.in_channels,
        )

        if mode == "generate":
            output = generate_long_video(
                api_fn=self.api_fn,
                sampling_option=local_opt,
                prompt=prompt,
                total_frames=int(total_frames),
                chunk_frames=int(chunk_frames),
                overlap_frames=int(overlap_frames),
                fps=self.fps,
                adaptive_steps=True,
                seed=self.seed,
                **common_kwargs,
            )
        elif mode == "extend":
            if not source_video:
                raise gr.Error("Please provide a source video for extend mode.")
            output = extend_video(
                api_fn=self.api_fn,
                sampling_option=local_opt,
                source_video_path=source_video,
                continue_prompt=prompt,
                extend_frames=int(extend_frames),
                chunk_frames=int(chunk_frames),
                overlap_frames=int(overlap_frames),
                fps=self.fps,
                seed=self.seed,
                **common_kwargs,
            )
        elif mode == "remaster":
            if not source_video:
                raise gr.Error("Please provide a source video for remaster mode.")
            output = remaster_video(
                api_fn=self.api_fn,
                sampling_option=local_opt,
                source_video_path=source_video,
                remaster_prompt=prompt,
                fps=self.fps,
                denoise_strength=float(denoise_strength),
                **common_kwargs,
            )
        else:
            raise gr.Error(f"Unknown mode: {mode}")

        output_dir = tempfile.mkdtemp(prefix="opensora_long_form_ui_")
        output_path = os.path.join(output_dir, "result")
        save_sample(output[0].cpu(), save_path=output_path, fps=self.fps, force_video=True, verbose=False)
        return f"{output_path}.mp4"


def build_ui(runtime: Runtime):
    with gr.Blocks(title="Open-Sora Long-Form UI") as demo:
        gr.Markdown("## Open-Sora Long-Form / Extend / Remaster")
        gr.Markdown("Load once, then run on localhost as an alternative to CLI commands.")

        with gr.Row():
            mode = gr.Dropdown(choices=["generate", "extend", "remaster"], value="generate", label="Mode")
            source_video = gr.Video(label="Source Video (required for extend/remaster)", type="filepath")

        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the scene/continuation/remaster intent")

        with gr.Row():
            total_frames = gr.Slider(64, 2048, value=720, step=1, label="Total Frames (generate)")
            extend_frames = gr.Slider(32, 1536, value=360, step=1, label="Extend Frames (extend)")

        with gr.Row():
            chunk_frames = gr.Slider(33, 257, value=97, step=1, label="Chunk Frames")
            overlap_frames = gr.Slider(0, 64, value=16, step=1, label="Overlap Frames")
            denoise_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Denoise Strength (remaster)")

        with gr.Row():
            num_steps = gr.Slider(10, 120, value=40, step=1, label="Sampling Steps")
            guidance = gr.Slider(1.0, 12.0, value=7.0, step=0.1, label="Guidance")
            guidance_img = gr.Slider(1.0, 12.0, value=4.0, step=0.1, label="Image Guidance")

        run_btn = gr.Button("Run")
        output_video = gr.Video(label="Generated Video")

        run_btn.click(
            fn=runtime.run,
            inputs=[
                mode,
                prompt,
                source_video,
                total_frames,
                extend_frames,
                chunk_frames,
                overlap_frames,
                denoise_strength,
                num_steps,
                guidance,
                guidance_img,
            ],
            outputs=[output_video],
        )

    return demo


def main():
    args = parse_args()
    runtime = Runtime(args.config)
    demo = build_ui(runtime)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
