# Lab 02 — Local LLM API Client (LM Studio)

## Reproducible execution (Windows 11 / Python 3.13)

Run from the repository root in PowerShell. Each lab has an isolated `.venv` and a complete `requirements.lock.txt`.

```powershell
.\setup.ps1 -Lab 2
.\run.ps1 -Lab 2 -Smoke
.\run.ps1 -Lab 2
```

LM Studio: http://localhost:1234/v1, Ministral 3 3B, context 16384, GPU offload disabled.

`-Cuda` selects PyTorch 2.14.1+cu130 from the official CUDA 13.0 wheel index. The NVIDIA driver is required; these wheels do not require a separate CUDA toolkit. Omit this switch for CPU PyTorch. Vision smoke tests use the original model weights, batch size 1, 512x512 images and short real inference. They do not establish image quality or repeat the full experimental matrix.

Model revision SHAs and weight checksums are in the root `models.lock.json`. Distributed input checksums are in `inputs.sha256.json` (text is normalized to LF). Setup downloads required models and NNTS datasets. Smoke tests use real services; unavailable services cause a failure. See the root `REPRODUCIBILITY.md` for sources, service preparation and verification limits.

## Overview

Interactive multi-turn CLI client integrating a local instruction-tuned LLM served by LM Studio (`mistralai/ministral-3-3b`) through the OpenAI-compatible Chat Completions API (`openai` Python SDK). Supports configurable system prompts, temperature, max completion tokens, endpoint `/v1/models` preflight verification, and Markdown dialog transcript logging.

## Main Implementation Files

- [`src/client.py`](src/client.py) — client configuration, `/v1/models` availability check, and Chat Completions request wrapper
- [`src/cli.py`](src/cli.py) — `argparse` CLI entrypoint, interactive turn loop, Markdown transcript persistence under `outputs/dialogs/`
- [`requirements.txt`](requirements.txt) — pinned `openai` SDK client dependency

## How to Run

1. Install dependencies:

   ```powershell
   ```

2. Start LM Studio's Local Server at `http://localhost:1234/v1` with `mistralai/ministral-3-3b` loaded and verify connectivity:

   ```powershell
   curl.exe http://localhost:1234/v1/models
   ```

3. Run the interactive CLI (defaults: `--base-url http://localhost:1234/v1`, `--model mistralai/ministral-3-3b`):

   ```powershell
   .\.venv\Scripts\python.exe -m labs.lab02_llm_api.src.cli `
       --system-prompt "You are a concise machine learning tutor." `
       --temperature 0.2 `
       --max-tokens 400
   ```

## Representative Example

Inspect available CLI flags non-interactively via `--help`:

```powershell
.\.venv\Scripts\python.exe -m labs.lab02_llm_api.src.cli --help
```

During an interactive session, each prompt sends the full local conversation history to `POST /v1/chat/completions` and saves the completed transcript locally under `labs/lab02_llm_api/outputs/dialogs/`.
