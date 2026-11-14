# Lab 04 — Stable Diffusion v1.5 Text-to-Image Synthesis

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 4 -Cuda
.\run.ps1 -Lab 4 -Smoke
.\run.ps1 -Lab 4
```

No separate inference server is required.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

## Overview

Controlled latent text-to-image synthesis with `stable-diffusion-v1-5` (`runwayml/stable-diffusion-v1-5`, FP16 + `model_cpu_offload` on 4 GB CUDA VRAM). Explores initial Gaussian noise inspection, CLIP tokenization, PNDM iterative denoising progression, Classifier-Free Guidance (CFG) scale, step budgets, negative prompts, and stylistic prompting.

## Main Implementation Files

- [`notebooks/01_text_to_image.ipynb`](notebooks/01_text_to_image.ipynb) — interactive walkthrough running in offline `PORTFOLIO_MODE=replay` (over committed examples) or GPU `PORTFOLIO_MODE=live`
- [`src/run.py`](src/run.py) — CLI entrypoint for batch generation across experiment matrices (`python -m labs.lab04_stable_diffusion.src.run`)
- [`src/pipeline.py`](src/pipeline.py) & [`src/pipeline_inspection.py`](src/pipeline_inspection.py) — Diffusers pipeline setup, VRAM offload, tokenizer/UNet/VAE tensor shape inspection, and denoising progression inspection
- [`src/generation.py`](src/generation.py), [`src/experiments.py`](src/experiments.py), [`src/evaluation.py`](src/evaluation.py), [`src/visualization.py`](src/visualization.py) — deterministic seed generation, parameter grids, summary tables, and comparison grids
- [`outputs/examples/`](outputs/examples/) — 4 curated 512×512 PNGs, [`generations.csv`](outputs/examples/generations.csv), [`environment.json`](outputs/examples/environment.json), [`intermediate_inspection.json`](outputs/examples/intermediate_inspection.json), and [`visual_observations.json`](outputs/examples/visual_observations.json)

## How to Run

Install requirements and run the notebook offline (`PORTFOLIO_MODE=replay`) or with a CUDA GPU (`PORTFOLIO_MODE=live`):

```powershell
$env:PORTFOLIO_MODE = "replay"
.\.venv-lab04\Scripts\python.exe -m jupyter notebook labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb
```

Live generation CLI (requires CUDA GPU):

```powershell
.\.venv-lab04\Scripts\python.exe -m labs.lab04_stable_diffusion.src.run
```

## Representative Example

From [`outputs/examples/generations.csv`](outputs/examples/generations.csv) (NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM, FP16 `model_cpu_offload`):

- **Case**: `baseline` (`seed=42`, `num_inference_steps=30`, `guidance_scale=7.5`, `PNDMScheduler`, `512x512`)
- **Runtime**: `18.88 s`, peak CUDA allocation: `2373.07 MiB`
- **Output**: [`baseline_seed42_steps30_cfg7p5_b0126393ec9c8f32.png`](outputs/examples/baseline_seed42_steps30_cfg7p5_b0126393ec9c8f32.png)

![Baseline Stable Diffusion Generation](outputs/examples/baseline_seed42_steps30_cfg7p5_b0126393ec9c8f32.png)
