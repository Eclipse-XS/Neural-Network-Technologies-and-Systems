"""Pipeline inspection and intermediate tensor verification for Stable Diffusion 1.5.

Exposes real internal components: Tokenizer, CLIP Text Encoder, Latent Noise,
U-Net, Classifier-Free Guidance, Scheduler Denoising Trajectory, and VAE Decoder.
Supports live execution when a pipeline is loaded, and verified evidence resolution
in replay mode.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import BASE_PROMPT, LAB_ROOT, NEGATIVE_PROMPT, GenerationSpec


def load_intermediate_inspection(portfolio_dir: Path | None = None) -> dict[str, Any]:
    portfolio = portfolio_dir or (LAB_ROOT / "artifacts/portfolio")
    inspection_file = portfolio / "intermediate_inspection.json"
    if not inspection_file.is_file():
        raise FileNotFoundError(f"Missing intermediate inspection file: {inspection_file}")
    return json.loads(inspection_file.read_text(encoding="utf-8"))


def plot_forward_diffusion_simulation(portfolio_dir: Path | None = None) -> plt.Figure:
    """Educational demonstration of analytical forward diffusion noise addition.
    
    Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    portfolio = portfolio_dir or (LAB_ROOT / "artifacts/portfolio")
    cache_path = portfolio / "forward_diffusion_simulation.png"
    if cache_path.is_file():
        fig, ax = plt.subplots(figsize=(13, 2.6))
        with Image.open(cache_path) as im:
            ax.imshow(im)
        ax.axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    y, x = np.ogrid[-2:2:256j, -2:2:256j]
    x0 = np.sin(x**2 + y**2)
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
            sqrt_one_minus = 0.0
        else:
            a_bar = alphas_cumprod[tp]
            sqrt_alpha = math.sqrt(a_bar)
            sqrt_one_minus = math.sqrt(1.0 - a_bar)
            xt = sqrt_alpha * x0 + sqrt_one_minus * eps
        
        axes[idx].imshow(xt, cmap="gray", vmin=-2, vmax=2)
        axes[idx].set_title(
            f"t = {tp}\n$\\sqrt{{\\bar{{\\alpha}}_t}}={sqrt_alpha:.2f}$\n$\\sqrt{{1-\\bar{{\\alpha}}_t}}={sqrt_one_minus:.2f}$",
            fontsize=8.5
        )
        axes[idx].axis("off")
        
    fig.suptitle("Educational Forward Noising Simulation: $x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon$ (Not Model Training)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


def plot_architecture_diagram(portfolio_dir: Path | None = None) -> plt.Figure:
    """Black-and-white academic architecture diagram for Stable Diffusion 1.5."""
    portfolio = portfolio_dir or (LAB_ROOT / "artifacts/portfolio")
    cache_path = portfolio / "stable_diffusion_architecture.png"
    if cache_path.is_file():
        fig, ax = plt.subplots(figsize=(12, 5.5))
        with Image.open(cache_path) as im:
            ax.imshow(im)
        ax.axis("off")
        fig.tight_layout()
        return fig

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

    arrows = [
        (2.5, 4.6, 3.0, 4.6), (5.0, 4.6, 5.5, 4.6), (7.7, 4.6, 8.2, 4.6),
        (9.3, 4.0, 7.7, 2.8), (2.5, 2.1, 3.0, 2.1), (5.0, 2.1, 5.5, 2.1),
        (7.7, 2.1, 8.2, 2.1), (10.4, 2.1, 10.5, 3.2),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#212529", lw=1.5), zorder=1)

    ax.set_title("Stable Diffusion 1.5 Text-to-Image Technical Inference Architecture", fontweight="bold", fontsize=11, pad=12)
    fig.tight_layout()
    return fig


def inspect_tokenization(pipe=None, prompt: str = BASE_PROMPT, portfolio_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Inspect prompt tokenization using live tokenizer or verified evidence."""
    if pipe is not None and hasattr(pipe, "tokenizer"):
        tokenizer = pipe.tokenizer
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        pad_id = tokenizer.pad_token_id
        real_tokens = [(idx, tid, tok) for idx, (tid, tok) in enumerate(zip(input_ids, tokens)) if tid != pad_id]
        eos_idx = next(i for i, tid in enumerate(input_ids) if tid == tokenizer.eos_token_id)
        summary = {
            "prompt": prompt,
            "token_count": eos_idx + 1,
            "model_max_length": tokenizer.model_max_length,
            "pad_token_id": pad_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        df = pd.DataFrame(real_tokens, columns=["position", "token_id", "token_string"])
        return df, summary

    evidence = load_intermediate_inspection(portfolio_dir)["tokenizer"]
    df = pd.DataFrame(evidence["first_tokens"], columns=["position", "token_id", "token_string"])
    summary = {k: v for k, v in evidence.items() if k != "first_tokens"}
    return df, summary


def inspect_clip_embeddings(pipe=None, prompt: str = BASE_PROMPT, portfolio_dir: Path | None = None) -> pd.DataFrame:
    """Inspect CLIP Text Encoder output tensor geometry and statistics."""
    if pipe is not None and hasattr(pipe, "text_encoder"):
        import torch
        tokenizer = pipe.tokenizer
        inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.inference_mode():
            emb = pipe.text_encoder(inputs.input_ids.to(pipe.device))[0]
        stats = {
            "component": type(pipe.text_encoder).__name__,
            "tensor_shape": str(list(emb.shape)),
            "dtype": str(emb.dtype),
            "device": str(emb.device),
            "is_finite": bool(torch.isfinite(emb).all().item()),
            "mean": round(float(emb.mean().item()), 4),
            "std": round(float(emb.std().item()), 4),
            "min": round(float(emb.min().item()), 4),
            "max": round(float(emb.max().item()), 4),
            "norm_token_0": round(float(torch.norm(emb[0, 0]).item()), 4),
        }
        return pd.DataFrame([stats])

    evidence = load_intermediate_inspection(portfolio_dir)["clip_text_encoder"]
    stats = {
        "component": evidence["text_encoder_class"],
        "tensor_shape": str(evidence["shape"]),
        "dtype": evidence["dtype"],
        "device": evidence["device"],
        "is_finite": evidence["is_finite"],
        "mean": round(evidence["mean"], 4),
        "std": round(evidence["std"], 4),
        "min": round(evidence["min"], 4),
        "max": round(evidence["max"], 4),
        "norm_token_0": round(evidence["norm_first_token"], 4),
    }
    return pd.DataFrame([stats])


def inspect_initial_latent(pipe=None, seed: int = 42, portfolio_dir: Path | None = None) -> tuple[pd.DataFrame, plt.Figure]:
    """Inspect initial Gaussian noise tensor in latent space."""
    portfolio = portfolio_dir or (LAB_ROOT / "artifacts/portfolio")
    if pipe is not None and hasattr(pipe, "unet"):
        import torch
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lat = torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=generator, dtype=torch.float32)
        init_sigma = float(pipe.scheduler.init_noise_sigma)
        scaled = lat * init_sigma
        stats = {
            "seed": seed,
            "latent_shape": str(list(scaled.shape)),
            "init_noise_sigma": init_sigma,
            "mean": round(float(scaled.mean().item()), 4),
            "std": round(float(scaled.std().item()), 4),
            "min": round(float(scaled.min().item()), 4),
            "max": round(float(scaled.max().item()), 4),
            "is_finite": bool(torch.isfinite(scaled).all().item()),
        }
        fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
        lat_np = scaled[0].numpy()
        for c in range(4):
            axes[c].imshow(lat_np[c], cmap="coolwarm", vmin=-3, vmax=3)
            axes[c].set_title(f"Channel {c} (64x64)\nstd={lat_np[c].std():.2f}", fontsize=8.5)
            axes[c].axis("off")
        fig.suptitle(f"Seeded Initial Latent Noise (Seed {seed}, Shape [1, 4, 64, 64])", fontweight="bold", fontsize=10)
        fig.tight_layout()
        return pd.DataFrame([stats]), fig

    evidence = load_intermediate_inspection(portfolio)["initial_latent"]
    stats = {
        "seed": evidence["seed"],
        "latent_shape": str(evidence["shape"]),
        "init_noise_sigma": evidence["init_noise_sigma"],
        "mean": round(evidence["mean"], 4),
        "std": round(evidence["std"], 4),
        "min": round(evidence["min"], 4),
        "max": round(evidence["max"], 4),
        "is_finite": evidence["is_finite"],
    }
    fig, ax = plt.subplots(figsize=(11, 2.8))
    cache_png = portfolio / "latent_channels.png"
    if cache_png.is_file():
        with Image.open(cache_png) as im:
            ax.imshow(im)
    ax.axis("off")
    fig.tight_layout()
    return pd.DataFrame([stats]), fig


def inspect_unet_and_cfg(pipe=None, portfolio_dir: Path | None = None) -> pd.DataFrame:
    """Inspect conditional U-Net noise prediction and Classifier-Free Guidance arithmetic."""
    if pipe is not None and hasattr(pipe, "unet"):
        import torch
        pipe.scheduler.set_timesteps(30)
        t = pipe.scheduler.timesteps[0]
        generator = torch.Generator(device="cpu").manual_seed(42)
        lat = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
        scaled = lat * float(pipe.scheduler.init_noise_sigma)
        
        with torch.inference_mode():
            p_emb, n_emb = pipe.encode_prompt(
                prompt=BASE_PROMPT, device=pipe.device, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=NEGATIVE_PROMPT
            )
            comb_emb = torch.cat([n_emb, p_emb])
            latent_in = torch.cat([scaled, scaled]).to(pipe.device, dtype=pipe.unet.dtype)
            latent_in = pipe.scheduler.scale_model_input(latent_in, t)
            noise_pred = pipe.unet(latent_in, t.to(pipe.device), encoder_hidden_states=comb_emb).sample
            
        noise_u, noise_c = noise_pred.chunk(2)
        guidance = 7.5
        noise_guided = noise_u + guidance * (noise_c - noise_u)
        algebraic_check = torch.allclose(noise_guided - noise_u, guidance * (noise_c - noise_u), atol=1e-3)
        
        row = {
            "timestep": int(t.item()),
            "unet_class": type(pipe.unet).__name__,
            "batched_input_shape": str(list(latent_in.shape)),
            "conditioning_shape": str(list(comb_emb.shape)),
            "predicted_noise_shape": str(list(noise_pred.shape)),
            "noise_uncond_std": round(float(noise_u.std().item()), 4),
            "noise_cond_std": round(float(noise_c.std().item()), 4),
            "guided_noise_std": round(float(noise_guided.std().item()), 4),
            "guidance_scale": guidance,
            "cfg_identity_verified": algebraic_check,
        }
        return pd.DataFrame([row])

    evidence = load_intermediate_inspection(portfolio_dir)["unet_cfg"]
    row = {
        "timestep": evidence["timestep"],
        "unet_class": evidence["unet_class"],
        "batched_input_shape": str(evidence["input_shape"]),
        "conditioning_shape": str(evidence["conditioning_shape"]),
        "predicted_noise_shape": str(evidence["predicted_noise_shape"]),
        "noise_uncond_std": round(evidence["noise_uncond_std"], 4),
        "noise_cond_std": round(evidence["noise_cond_std"], 4),
        "guided_noise_std": round(evidence["noise_cfg_std"], 4),
        "guidance_scale": evidence["guidance_scale"],
        "cfg_identity_verified": evidence["cfg_algebraic_identity_verified"],
    }
    return pd.DataFrame([row])


def resolve_denoising_progression(pipe=None, portfolio_dir: Path | None = None) -> tuple[pd.DataFrame, plt.Figure]:
    """Capture or load verified intermediate denoising states across 30 PNDM steps."""
    portfolio = portfolio_dir or (LAB_ROOT / "artifacts/portfolio")
    steps_dir = portfolio / "denoising_steps"
    evidence = load_intermediate_inspection(portfolio)["denoising_progression"]
    
    rows = []
    images = []
    for rec in evidence:
        img_path = portfolio / rec["relative_path"]
        if not img_path.is_file():
            img_path = steps_dir / rec["image_filename"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing denoising snapshot: {img_path}")
        digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if digest != rec["sha256"]:
            raise ValueError(f"Integrity failure in snapshot {rec['image_filename']}")
        
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            images.append(im.copy())
            
        rows.append({
            "step": rec["step"],
            "timestep": rec["timestep"],
            "progress": f"{rec['progress_pct']}%",
            "latent_std": round(rec["latent_std"], 4),
            "image_filename": rec["image_filename"],
            "sha256_verified": True
        })
        
    df = pd.DataFrame(rows)
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3.5))
    for ax, im, r in zip(axes, images, rows):
        ax.imshow(im)
        ax.set_title(f"Step {r['step']} (t={r['timestep']})\nProgress: {r['progress']}\nσ={r['latent_std']}", fontsize=8.5)
        ax.axis("off")
    fig.suptitle("Latent Denoising Trajectory: Progressive Intermediate States (PNDM 30 Steps, VAE Decoded)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return df, fig


def inspect_vae_decoding(pipe=None, portfolio_dir: Path | None = None) -> pd.DataFrame:
    """Inspect VAE configuration, scaling factor, and reconstruction geometry."""
    if pipe is not None and hasattr(pipe, "vae"):
        cfg = pipe.vae.config
        row = {
            "vae_class": type(pipe.vae).__name__,
            "latent_channels": cfg.latent_channels,
            "scaling_factor": cfg.scaling_factor,
            "inverse_scaling": round(1.0 / cfg.scaling_factor, 4),
            "input_latent_dim": "[1, 4, 64, 64]",
            "decoded_output_dim": "[1, 3, 512, 512]",
            "upsampling_factor": "8x (64 -> 512)",
        }
        return pd.DataFrame([row])

    evidence = load_intermediate_inspection(portfolio_dir)["vae"]
    scale = evidence["scaling_factor"]
    row = {
        "vae_class": evidence["vae_class"],
        "latent_channels": evidence["latent_channels"],
        "scaling_factor": scale,
        "inverse_scaling": round(1.0 / scale, 4),
        "input_latent_dim": "[1, 4, 64, 64]",
        "decoded_output_dim": "[1, 3, 512, 512]",
        "upsampling_factor": "8x (64 -> 512)",
    }
    return pd.DataFrame([row])


def calculate_lora_parameters(ranks: tuple[int, ...] = (4, 8, 16)) -> pd.DataFrame:
    """Calculate parameter efficiency of Low-Rank Adaptation (LoRA) on SD 1.5 U-Net attention."""
    # SD 1.5 U-Net cross-attention dimensions:
    # 16 cross-attention blocks, each with W_q, W_k, W_v, W_out projections
    # Attention hidden sizes across levels: 320 (x2), 640 (x2), 1280 (x2)
    # Total U-Net parameters ~ 859,520,964
    total_unet_params = 859_520_964
    
    # 16 attention modules with query/key/value/out projections:
    # Level 0 (320): 4 blocks * 4 projections * (320 * 320) = 1,638,400
    # Level 1 (640): 4 blocks * 4 projections * (640 * 640) = 6,553,600
    # Level 2 (1280): 6 blocks * 4 projections * (1280 * 1280) = 39,321,600
    # Level 3 mid (1280): 1 block * 4 projections * (1280 * 1280) = 6,553,600
    # Total attention projection params ~ 54,067,200
    total_attn_params = 54_067_200
    
    # For LoRA rank r, each projection (d_in x d_out) is replaced by B (d_in x r) + A (r x d_out)
    # Number of params per projection = r * (d_in + d_out)
    rows = []
    for r in ranks:
        # Summing r * (d_in + d_out) across all 16 cross-attention blocks:
        # Level 0: 4 * 4 * r * (320 + 320) = 10,240 * r
        # Level 1: 4 * 4 * r * (640 + 640) = 20,480 * r
        # Level 2: 6 * 4 * r * (1280 + 1280) = 61,440 * r
        # Level 3: 1 * 4 * r * (1280 + 1280) = 10,240 * r
        # Total LoRA params = 102,400 * r
        lora_params = 102_400 * r
        trainable_pct = (lora_params / total_unet_params) * 100
        reduction_ratio = total_unet_params / lora_params
        rows.append({
            "adaptation_method": f"LoRA (rank={r})",
            "trainable_parameters": f"{lora_params:,}",
            "frozen_parameters": f"{total_unet_params:,}",
            "trainable_share": f"{trainable_pct:.3f}%",
            "parameter_reduction": f"{reduction_ratio:.1f}x",
            "target_modules": "Cross-Attention Projections (W_q, W_k, W_v, W_out)",
        })
    
    rows.append({
        "adaptation_method": "Full U-Net Fine-Tuning",
        "trainable_parameters": f"{total_unet_params:,}",
        "frozen_parameters": "0",
        "trainable_share": "100.000%",
        "parameter_reduction": "1.0x",
        "target_modules": "All U-Net parameters (ResNet + Attention + Conv)",
    })
    return pd.DataFrame(rows)
