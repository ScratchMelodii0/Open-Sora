save_dir = "samples_long_form"
seed = 42
batch_size = 1
dtype = "bf16"
fps_save = 24

# prompt for generate / extend / remaster
prompt = "A single-take cinematic tracking shot in Tokyo at night, rain reflections, neon signs, a woman in a yellow coat walks through the street, realistic lens effects and stable identity."

# base sampling options (per chunk)
sampling_option = dict(
    resolution="256px",
    aspect_ratio="16:9",
    num_frames=97,
    num_steps=40,
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    guidance=7.0,
    guidance_img=4.0,
    text_osci=True,
    image_osci=False,
    scale_temporal_osci=False,
    method="i2v",
    seed=None,
)

long_form = dict(
    mode="generate",  # generate | extend | remaster
    total_frames=720,  # 30 seconds at 24 fps
    extend_frames=360,
    chunk_frames=97,
    overlap_frames=16,
    adaptive_steps=True,
    extension_prompt="Continue the scene with the same woman and exact visual style; she enters an alley and a tram passes behind her.",
    source_video="samples/source.mp4",
    denoise_strength=0.35,
    output_name="long_form_demo",
)

model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
)
t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai/clip-vit-large-patch14",
    max_length=77,
)
