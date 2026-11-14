# Lab 05 — Multimodal Visual Question Answering (LLaVA)

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 5 -Cuda
.\run.ps1 -Lab 5 -Smoke
.\run.ps1 -Lab 5
```

No separate inference server is required.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

## Overview

Single-image Visual Question Answering (VQA) evaluation of `llava-hf/llava-interleave-qwen-0.5b-hf` (SigLIP vision encoder + 2-layer MLP projector + Qwen2 0.5B backbone) in FP16 on 4 GB CUDA VRAM. Investigates factual grounding, decoding temperature stability, `max_new_tokens` truncation, paraphrasing sensitivity, and unanswerable visual queries across a dual CPU-replay / CUDA-live workflow.

## Main Implementation Files

- Notebooks:
  - [`notebooks/01_llava_vqa.ipynb`](notebooks/01_llava_vqa.ipynb) — offline CPU walkthrough and analysis (`LAB05_MODE=REPLAY`) or full CUDA run (`LAB05_MODE=LIVE`)
  - [`notebooks/02_llava_live_inference.ipynb`](notebooks/02_llava_live_inference.ipynb) — interactive single-image CUDA practical notebook
- Source modules (`src/`):
  - [`src/config.py`](src/config.py) — paths, model ID/revision, image SHA-256 constant
  - [`src/model.py`](src/model.py) — Hugging Face Transformers pipeline loading and CUDA environment checks
  - [`src/inference.py`](src/inference.py) & [`src/live.py`](src/live.py) — cache-aware VQA execution and unreviewed live GPU runner
  - [`src/experiments.py`](src/experiments.py), [`src/evidence.py`](src/evidence.py), [`src/evaluation.py`](src/evaluation.py), [`src/visualization.py`](src/visualization.py) — experiment definitions, schema/hash checks, and plotting
  - [`src/run.py`](src/run.py) — CLI batch runner (`python -m labs.lab05_multimodal.src.run`)
- Assets & Examples:
  - [`questions.json`](questions.json) — 10 structured VQA questions (descriptive, factual, analytical, unanswerable)
  - [`assets/input/primary_image.png`](assets/input/primary_image.png) & [`assets/input/source.json`](assets/input/source.json)
  - [`outputs/examples/vqa_examples.json`](outputs/examples/vqa_examples.json) & [`outputs/examples/review_labels.json`](outputs/examples/review_labels.json)

## How to Run

Offline replay notebook (works on CPU without downloading model weights):

```powershell
$env:LAB05_MODE = "REPLAY"
.\labs\lab05_multimodal\.venv\Scripts\python.exe -m jupyter notebook labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb
```

Live CUDA CLI runner (requires CUDA GPU and Hugging Face model access):

```powershell
.\labs\lab05_multimodal\.venv\Scripts\python.exe -m labs.lab05_multimodal.src.run
```

## Representative Example

From [`outputs/examples/vqa_examples.json`](outputs/examples/vqa_examples.json) (`llava-hf/llava-interleave-qwen-0.5b-hf`, FP16 on `cuda:0`):

- **`fact_01`**: *"Is the cabin on the left, in the center, or on the right side of the image?"* -> `"left"` (`2` tokens, EOS=`True`, `2.08 s`)
- **`desc_01`**: *"Describe the main scene in this image in two sentences."* -> `"The image shows a cabin nestled among trees on a lake, with a reflection of the cabin in the water. The cabin has a wooden roof and a small deck. The surrounding environment is lush with green trees and mountains in the background."` (`48` tokens, EOS=`True`, `2.79 s`)

![Canonical VQA Input Image](assets/input/primary_image.png)
