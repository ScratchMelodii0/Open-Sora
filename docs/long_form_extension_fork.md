# Open-Sora Long-Form & Extension Fork

This fork adds a practical long-form pipeline on top of the existing Open-Sora inference stack without breaking base APIs.

## What is added

- **Chunked autoregressive long-video generation** via `generate_long_video()`:
  - First chunk is generated with `t2v`.
  - Next chunks are generated with `v2v_head` and stitched using overlap cross-fading.
  - Optional adaptive denoising-step schedule for later chunks.
- **Video extension mode** via `extend_video()`:
  - Reads an existing clip and appends continuation generated with the same style/motion continuity.
- **Video remaster mode** via `remaster_video()`:
  - Uses source clip as strong v2v condition and re-synthesizes details, then blends with source for reduced flicker.
- **A dedicated script**: `scripts/diffusion/long_form_inference.py`.
- **A reference config**: `configs/diffusion/inference/long_form_30s.py`.

## Design Notes

### Temporal strategy

1. Generate in chunks (`chunk_frames`) to keep memory bounded.
2. Preserve continuity with strong visual conditioning (`v2v_head`) from the previous chunk tail.
3. Blend overlaps (`overlap_frames`) in pixel space to smooth transitions.
4. Increase denoising steps on later chunks (optional) to reduce long-horizon drift.

### Model compatibility

No architectural checkpoint conversion is required. The implementation reuses:

- `prepare_models()` and `prepare_api()` from `opensora.utils.sampling`.
- Existing i2v/v2v condition path in `opensora.utils.inference`.

## Usage

```bash
python scripts/diffusion/long_form_inference.py configs/diffusion/inference/long_form_30s.py
```

Change `long_form.mode` to one of:

- `generate` (text-to-long-video)
- `extend` (continue an existing video)
- `remaster` (video refinement)


## Localhost UI (Gradio)

A UI alternative to CLI commands is available at `gradio/long_form_app.py`.

```bash
python gradio/long_form_app.py --config configs/diffusion/inference/long_form_30s.py --host 127.0.0.1 --port 7861
```

Then open `http://127.0.0.1:7861` in your browser.

The UI supports:
- `generate` (text-to-long-video)
- `extend` (source video + continuation prompt)
- `remaster` (source video + enhancement prompt)

