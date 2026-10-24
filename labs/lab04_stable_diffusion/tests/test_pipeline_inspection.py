from pathlib import Path
import pytest
from labs.lab04_stable_diffusion.src.config import LAB_ROOT
from labs.lab04_stable_diffusion.src.pipeline_inspection import (
    calculate_lora_parameters,
    inspect_clip_embeddings,
    inspect_initial_latent,
    inspect_tokenization,
    inspect_unet_and_cfg,
    inspect_vae_decoding,
    load_intermediate_inspection,
    plot_architecture_diagram,
    plot_forward_diffusion_simulation,
    resolve_denoising_progression,
)


def test_intermediate_inspection_schema_and_contents():
    data = load_intermediate_inspection()
    expected_keys = {
        "model_id", "revision", "dtype", "tokenizer", "clip_text_encoder",
        "initial_latent", "unet_cfg", "denoising_progression", "vae"
    }
    assert expected_keys.issubset(data.keys())
    assert data["model_id"] == "stable-diffusion-v1-5/stable-diffusion-v1-5"


def test_tokenization_and_clip_geometry():
    df_tokens, summary = inspect_tokenization()
    assert len(df_tokens) > 10
    assert {"position", "token_id", "token_string"}.issubset(df_tokens.columns)
    assert summary["model_max_length"] == 77
    assert summary["token_count"] <= 77

    df_clip = inspect_clip_embeddings()
    assert df_clip["tensor_shape"].iloc[0] == "[1, 77, 768]"
    assert df_clip["is_finite"].all()


def test_latent_and_unet_cfg_invariants():
    df_lat, fig_lat = inspect_initial_latent()
    assert df_lat["latent_shape"].iloc[0] == "[1, 4, 64, 64]"
    assert df_lat["is_finite"].all()

    df_unet_cfg = inspect_unet_and_cfg()
    assert df_unet_cfg["predicted_noise_shape"].iloc[0] == "[2, 4, 64, 64]"
    assert df_unet_cfg["cfg_identity_verified"].all()


def test_denoising_progression_and_vae():
    df_prog, fig_prog = resolve_denoising_progression()
    assert len(df_prog) == 5
    assert df_prog["sha256_verified"].all()

    df_vae = inspect_vae_decoding()
    assert df_vae["scaling_factor"].iloc[0] == 0.18215
    assert df_vae["decoded_output_dim"].iloc[0] == "[1, 3, 512, 512]"


def test_lora_parameter_reduction():
    df_lora = calculate_lora_parameters(ranks=(4, 8, 16))
    assert len(df_lora) == 4
    # All LoRA ranks should achieve over 99% parameter reduction
    rank_rows = df_lora.iloc[:3]
    for _, row in rank_rows.iterrows():
        pct = float(row["trainable_share"].rstrip("%"))
        assert pct < 1.0  # < 1% of total parameters
