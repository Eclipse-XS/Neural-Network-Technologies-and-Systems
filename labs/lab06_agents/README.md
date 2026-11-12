# Lab 06 — Autonomous LLM Agent with OpenAI Agents SDK

## Overview

Tool-using ReAct agent built with the OpenAI Agents SDK (`openai-agents`) against a local LM Studio server (`http://localhost:1234/v1`, `mistralai/ministral-3-3b`). Combines bounded local workspace tools (`list_workspace_files`, `search_workspace`, `read_workspace_file`), AST-sandboxed arithmetic (`calculate`), and isolated `SQLiteSession` multi-turn memory, comparing tool-enabled vs. tool-disabled agent runs.

## Main Implementation Files

- [`notebooks/01_openai_agents_sdk.ipynb`](notebooks/01_openai_agents_sdk.ipynb) — walkthrough covering AST safety checks, bounded workspace tools, `SQLiteSession` persistence, and saved trace inspection (`LAB06_RUN_LIVE_SMOKE=1` for live LM Studio probes)
- Source modules (`src/`):
  - [`src/config.py`](src/config.py) — paths, LM Studio URL/model config, workspace hash manifest settings
  - [`src/preflight.py`](src/preflight.py) — `/v1/models` endpoint and tool-calling preflight check
  - [`src/tools.py`](src/tools.py) — sandboxed workspace file tools (`assets/workspace/`) and AST math evaluator (`calculate`)
  - [`src/agent.py`](src/agent.py) — OpenAI Agents SDK `Agent` and `OpenAIChatCompletionsModel` factory
  - [`src/runner.py`](src/runner.py) — SDK `Runner` execution loop, token/tool trace logger, and output sanitizer
  - [`src/experiments.py`](src/experiments.py) & [`src/evaluation.py`](src/evaluation.py) — scenario runner (`python -m labs.lab06_agents.src.experiments`) and trace loader
- Assets & Examples:
  - [`scenarios.json`](scenarios.json) — scenario definitions (`lookup`, `multi_step`, `unsupported`, `comparison`, `memory`)
  - [`assets/workspace/`](assets/workspace/) — local copies of `lab03_report.md`, `lab04_report.md`, `lab05_report.md`, and `manifest.json`
  - [`outputs/runs/runs.jsonl`](outputs/runs/runs.jsonl), [`outputs/metadata/latest_batch.json`](outputs/metadata/latest_batch.json), and [`outputs/traces/`](outputs/traces/) — 4 curated SDK execution traces

## How to Run

1. Open the notebook offline to verify tools, SQLite memory isolation, and saved execution traces:

   ```powershell
   $env:LAB06_RUN_LIVE_SMOKE = "0"
   $env:LAB06_RUN_FULL_LIVE = "0"
   .\.venv-lab06\Scripts\python.exe -m jupyter notebook labs/lab06_agents/notebooks/01_openai_agents_sdk.ipynb
   ```

2. Run live scenario batches against LM Studio (`http://localhost:1234/v1` with `mistralai/ministral-3-3b`):

   ```powershell
   .\.venv-lab06\Scripts\python.exe -m labs.lab06_agents.src.experiments
   ```

## Representative Example

From [`outputs/runs/runs.jsonl`](outputs/runs/runs.jsonl) and trace [`0dcaa0248fca482d84d0cdac2f07ddca.json`](outputs/traces/0dcaa0248fca482d84d0cdac2f07ddca.json):

- **Scenario (`lab4`, `with_tools`)**: *"Which Stable Diffusion model was used in Lab 4?"*
- **Tool Calls**: `list_workspace_files` -> `search_workspace` (`2` calls, `0` errors, `3` requests, `3458` tokens, `9.78 s`)
- **Output**: Identifies **`stable-diffusion-v1-5`** citing line `138` of `lab04_report.md`.
