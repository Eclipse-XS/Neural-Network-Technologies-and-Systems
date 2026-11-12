# Lab 04 — Stable Diffusion v1.5 Text-to-Image Synthesis

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
