# Lab 02 — Local LLM API Client (LM Studio)

## Overview

Interactive multi-turn CLI client integrating a local instruction-tuned LLM served by LM Studio (`mistralai/ministral-3-3b`) through the OpenAI-compatible Chat Completions API (`openai` Python SDK). Supports configurable system prompts, temperature, max completion tokens, endpoint `/v1/models` preflight verification, and Markdown dialog transcript logging.

## Main Implementation Files

- [`src/client.py`](src/client.py) — client configuration, `/v1/models` availability check, and Chat Completions request wrapper
- [`src/cli.py`](src/cli.py) — `argparse` CLI entrypoint, interactive turn loop, Markdown transcript persistence under `outputs/dialogs/`
- [`requirements.txt`](requirements.txt) — pinned `openai` SDK client dependency

## How to Run

1. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r labs/lab02_llm_api/requirements.txt
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
