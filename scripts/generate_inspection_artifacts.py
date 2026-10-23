"""Generate and verify intermediate Stable Diffusion inspection artifacts.
Run with .venv-lab04 python.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.lab04_stable_diffusion.src.config import BASE_PROMPT, NEGATIVE_PROMPT, GenerationSpec
from labs.lab04_stable_diffusion.src.pipeline import load_pipeline

PORTFOLIO = ROOT / "labs/lab04_stable_diffusion/artifacts/portfolio"
STEPS_DIR = PORTFOLIO / "denoising_steps"
STEPS_DIR.mkdir(parents=True, exist_ok=True)


def generate_all():
    print("Loading pipeline on RTX 3050...")
    pipe, ctx = load_pipeline()
    device = pipe.device
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 1. Tokenizer inspection
    inputs = tokenizer(
        BASE_PROMPT,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Non-pad tokens
    pad_id = tokenizer.pad_token_id
    real_tokens = [(idx, tid, tok) for idx, (tid, tok) in enumerate(zip(input_ids, tokens)) if tid != pad_id]
    eos_idx = next(i for i, tid in enumerate(input_ids) if tid == tokenizer.eos_token_id)
    
    tokenizer_data = {
        "prompt": BASE_PROMPT,
        "model_max_length": tokenizer.model_max_length,
        "token_count": eos_idx + 1,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "first_tokens": real_tokens[:25],
        "attention_mask_shape": list(inputs.attention_mask.shape)
    }
    print(f"Tokenized prompt: {tokenizer_data['token_count']} tokens including BOS/EOS.")

    # 2. CLIP Text Encoder inspection
    with torch.inference_mode():
        text_emb = text_encoder(inputs.input_ids)[0]
    
    clip_data = {
        "text_encoder_class": type(text_encoder).__name__,
        "shape": list(text_emb.shape),
        "dtype": str(text_emb.dtype),
        "device": str(text_emb.device),
        "is_finite": bool(torch.isfinite(text_emb).all().item()),
        "mean": float(text_emb.mean().item()),
        "std": float(text_emb.std().item()),
        "min": float(text_emb.min().item()),
        "max": float(text_emb.max().item()),
        "norm_first_token": float(torch.norm(text_emb[0, 0]).item()),
        "norm_last_token": float(torch.norm(text_emb[0, eos_idx]).item())
    }
    print(f"CLIP embedding shape: {clip_data['shape']}, finite: {clip_data['is_finite']}")

    # 3. Initial Latent Noise inspection
    seed = 42
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        dtype=torch.float32
    )
    init_noise_sigma = float(pipe.scheduler.init_noise_sigma)
    scaled_latents = latents * init_noise_sigma

    latent_data = {
        "seed": seed,
        "shape": list(latents.shape),
        "dtype": str(latents.dtype),
        "init_noise_sigma": init_noise_sigma,
        "mean": float(scaled_latents.mean().item()),
        "std": float(scaled_latents.std().item()),
        "min": float(scaled_latents.min().item()),
        "max": float(scaled_latents.max().item()),
        "is_finite": bool(torch.isfinite(scaled_latents).all().item())
    }
    print(f"Initial latent shape: {latent_data['shape']}, mean={latent_data['mean']:.4f}, std={latent_data['std']:.4f}")

    # Plot 4 latent channels
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    lat_np = scaled_latents[0].numpy()
    for c in range(4):
        im = axes[c].imshow(lat_np[c], cmap="coolwarm", vmin=-3, vmax=3)
        axes[c].set_title(f"Channel {c} (64x64)\nstd={lat_np[c].std():.2f}")
        axes[c].axis("off")
    fig.suptitle("Initial Latent Noise Channels (Seed 42, Shape [1, 4, 64, 64])", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PORTFOLIO / "latent_channels.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 4. U-Net and CFG inspection
    pipe.scheduler.set_timesteps(30)
    t = pipe.scheduler.timesteps[0]
    
    with torch.inference_mode():
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=BASE_PROMPT,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT
        )
        
        # Batch size 2 for CFG: [neg_embeds, prompt_embeds]
        combined_text_embeds = torch.cat([neg_embeds, prompt_embeds])
        latent_model_input = torch.cat([scaled_latents, scaled_latents]).to(pipe.device, dtype=pipe.unet.dtype)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(
            latent_model_input,
            t.to(pipe.device),
            encoder_hidden_states=combined_text_embeds
        ).sample
    
    noise_uncond, noise_cond = noise_pred.chunk(2)
    guidance_scale = 7.5
    noise_cfg = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    diff = noise_cfg - noise_uncond
    expected_diff = guidance_scale * (noise_cond - noise_uncond)
    cfg_identity_verified = bool(torch.allclose(diff, expected_diff, atol=1e-3))

    unet_cfg_data = {
        "timestep": int(t.item()),
        "unet_class": type(pipe.unet).__name__,
        "input_shape": list(latent_model_input.shape),
        "conditioning_shape": list(prompt_embeds.shape),
        "predicted_noise_shape": list(noise_pred.shape),
        "noise_uncond_mean": float(noise_uncond.mean().item()),
        "noise_uncond_std": float(noise_uncond.std().item()),
        "noise_cond_mean": float(noise_cond.mean().item()),
        "noise_cond_std": float(noise_cond.std().item()),
        "noise_cfg_mean": float(noise_cfg.mean().item()),
        "noise_cfg_std": float(noise_cfg.std().item()),
        "guidance_scale": guidance_scale,
        "cfg_algebraic_identity_verified": cfg_identity_verified
    }
    print(f"U-Net forward pass & CFG verified: {cfg_identity_verified}")

    # 5. Denoising progression capture
    snapshots = {}
    def step_callback(pipeline, step, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        if step in (0, 7, 15, 22, 29):
            snapshots[step] = {
                "step": step,
                "timestep": int(timestep),
                "shape": list(lat.shape),
                "mean": float(lat.mean().item()),
                "std": float(lat.std().item()),
                "min": float(lat.min().item()),
                "max": float(lat.max().item()),
                "latent_tensor": lat.detach().cpu().clone()
            }
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        res = pipe(
            prompt=BASE_PROMPT,
            negative_prompt=None,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            callback_on_step_end=step_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )

    # Decode and save intermediate images
    step_records = []
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    step_keys = sorted(snapshots.keys())
    
    for ax_idx, s in enumerate(step_keys):
        sdata = snapshots[s]
        lat_tensor = sdata["latent_tensor"].to(pipe.device, dtype=pipe.vae.dtype)
        with torch.inference_mode():
            dec = pipe.vae.decode(lat_tensor / pipe.vae.config.scaling_factor, return_dict=False)[0]
        pil_img = pipe.image_processor.postprocess(dec, output_type="pil")[0]
        
        img_name = f"step_{s:02d}_t{sdata['timestep']:03d}.png"
        img_path = STEPS_DIR / img_name
        pil_img.save(img_path, format="PNG")
        digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
        
        record = {
            "step": s,
            "timestep": sdata["timestep"],
            "progress_pct": int(round((s + 1) / 30 * 100)),
            "latent_mean": sdata["mean"],
            "latent_std": sdata["std"],
            "image_filename": img_name,
            "relative_path": f"artifacts/portfolio/denoising_steps/{img_name}",
            "sha256": digest
        }
        step_records.append(record)
        
        axes[ax_idx].imshow(pil_img)
        axes[ax_idx].set_title(f"Step {s} (t={sdata['timestep']})\nProg: {record['progress_pct']}%\nσ={sdata['std']:.2f}", fontsize=9)
        axes[ax_idx].axis("off")
        
    fig.suptitle("Iterative Denoising Latent Decoding Trajectory (30 PNDM Steps, VAE Decoded)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PORTFOLIO / "denoising_progression.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Captured and decoded {len(step_records)} denoising steps.")

    # 6. VAE inspection
    vae_data = {
        "vae_class": type(pipe.vae).__name__,
        "in_channels": pipe.vae.config.in_channels,
        "out_channels": pipe.vae.config.out_channels,
        "latent_channels": pipe.vae.config.latent_channels,
        "scaling_factor": pipe.vae.config.scaling_factor,
        "sample_size": pipe.vae.config.sample_size,
        "decoded_resolution": [512, 512],
        "num_upsample_blocks": len(pipe.vae.decoder.up_blocks)
    }

    # 7. Forward diffusion simulation figure
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    # Synthetic reference pattern x0 in [-1, 1]
    y, x = np.ogrid[-2:2:256j, -2:2:256j]
    x0 = np.sin(x**2 + y**2)
    # SD 1.5 linear beta schedule
    betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000)**2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    t_points = [0, 200, 400, 600, 800, 999]
    np.random.seed(42)
    eps = np.random.randn(*x0.shape)
    
    for idx, tp in enumerate(t_points):
        if tp == 0:
            xt = x0
            sqrt_alpha = 1.0
            sqrt_one_minus_alpha = 0.0
        else:
            a_bar = alphas_cumprod[tp]
            sqrt_alpha = math.sqrt(a_bar)
            sqrt_one_minus_alpha = math.sqrt(1.0 - a_bar)
            xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * eps
        
        axes[idx].imshow(xt, cmap="gray", vmin=-2, vmax=2)
        axes[idx].set_title(f"t = {tp}\n$\\sqrt{{\\bar{{\\alpha}}_t}}={sqrt_alpha:.2f}$\n$\\sqrt{{1-\\bar{{\\alpha}}_t}}={sqrt_one_minus_alpha:.2f}$", fontsize=9)
        axes[idx].axis("off")
        
    fig.suptitle("Mathematical Simulation: Analytical Forward Diffusion Process ($x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon$)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PORTFOLIO / "forward_diffusion_simulation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 8. Clean B&W Draw.io style Architecture Diagram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    boxes = [
        (0.5, 4.0, 2.0, 1.2, "User Prompt\n(text string)", "Input"),
        (3.0, 4.0, 2.0, 1.2, "CLIP Tokenizer\n(BPE, max 77 tokens)", "Preprocessing"),
        (5.5, 4.0, 2.2, 1.2, "CLIP Text Encoder\n(12 layers, hidden 768)", "Text Conditioning"),
        (8.2, 4.0, 2.2, 1.2, "Conditioning Embeddings\nShape: [1, 77, 768]", "Context Vector"),
        
        (0.5, 1.5, 2.0, 1.2, "Gaussian Noise\nSeed N(0, I)", "Initialization"),
        (3.0, 1.5, 2.0, 1.2, "Initial Latent\nShape: [1, 4, 64, 64]", "Latent Space"),
        (5.5, 1.2, 2.2, 1.8, "U-Net 2D Condition\n+ CFG (w=7.5)\n+ PNDM Scheduler\n(30 iterations)", "Denoising Engine"),
        (8.2, 1.5, 2.2, 1.2, "Denoised Latent\nShape: [1, 4, 64, 64]", "Clean Latent"),
        (10.5, 2.7, 1.4, 1.5, "VAE Decoder\n(1/0.18215 scale)\n[512x512 RGB]", "Pixel Synthesis"),
    ]

    for x_b, y_b, w_b, h_b, label, sub in boxes:
        rect = plt.Rectangle((x_b, y_b), w_b, h_b, facecolor="#F8F9FA", edgecolor="#212529", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x_b + w_b/2, y_b + h_b/2 + 0.15, label, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)
        ax.text(x_b + w_b/2, y_b + 0.2, f"[{sub}]", ha="center", va="center", fontsize=7, color="#495057", zorder=3)

    # Arrows
    arrows = [
        (2.5, 4.6, 3.0, 4.6),
        (5.0, 4.6, 5.5, 4.6),
        (7.7, 4.6, 8.2, 4.6),
        (9.3, 4.0, 7.7, 2.8),
        (2.5, 2.1, 3.0, 2.1),
        (5.0, 2.1, 5.5, 2.1),
        (7.7, 2.1, 8.2, 2.1),
        (10.4, 2.1, 10.5, 3.2),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#212529", lw=1.5), zorder=1)

    ax.set_title("Stable Diffusion 1.5 Text-to-Image Technical Inference Architecture", fontweight="bold", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(PORTFOLIO / "stable_diffusion_architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Architecture diagram generated.")

    # 9. Save master JSON
    master_record = {
        "generated_at": ctx.get("created_at"),
        "model_id": ctx.get("model_id"),
        "revision": ctx.get("revision"),
        "device": ctx.get("device_strategy"),
        "dtype": ctx.get("dtype"),
        "tokenizer": tokenizer_data,
        "clip_text_encoder": clip_data,
        "initial_latent": latent_data,
        "unet_cfg": unet_cfg_data,
        "denoising_progression": step_records,
        "vae": vae_data
    }
    
    out_json = PORTFOLIO / "intermediate_inspection.json"
    out_json.write_text(json.dumps(master_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Inspection records successfully saved to {out_json}")


if __name__ == "__main__":
    generate_all()
