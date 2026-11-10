# Reproducibility Matrix & Environment Verification Guide

This document defines the exact execution modes, required runtime dependencies, deterministic offline test commands, and live hardware/service reproduction commands across all seven laboratories (`lab01_transformer` through `lab07_final_project`).

---

## 1. Repository-Wide Reproducibility Matrix

| Lab | Domain | Offline / Replay Test Command | Live / Full Execution Command | Required External Services (Live Only) | Hardware Requirements | Committed Evidence Artifacts |
|:---:|:---|:---|:---|:---|:---|:---|
| **Lab 01** | Transformer Architectures (Encoder Classifier & Decoder LM) | `.\.venv-lab04\Scripts\python.exe -m pytest labs/lab01_transformer/tests -q` | `.\.venv-lab04\Scripts\python.exe labs/lab01_transformer/scripts/run_generation_diagnostics.py` | None (Public SMS Spam & BBC News CSVs verified by SHA-256) | CPU sufficient for tests; CUDA GPU recommended for full training | `labs/lab01_transformer/outputs/classification/sms_spam/` & `outputs/generation/bbc/` |
| **Lab 02** | Local LLM API Integration (OpenAI SDK + LM Studio) | `.\.venv-lab04\Scripts\python.exe -m pytest labs/lab02_llm_api/tests -q` | `.\.venv-lab04\Scripts\python.exe -m labs.lab02_llm_api.src.cli --model mistralai/ministral-3-3b` | LM Studio (`http://localhost:1234/v1`) | CPU for mocked tests; 4+ GB VRAM GPU for local LLM | `labs/lab02_llm_api/outputs/dialogs/*.md` |
| **Lab 03** | Redis Vector Store & Leakage-Free Hybrid RAG | `.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab03_rag/tests -q` | `.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab03_rag.src.run` | Redis Stack (`localhost:6379`) + LM Studio (`localhost:1234/v1`) | CPU for embedding & replay; 4+ GB VRAM for live LLM | `labs/lab03_rag/artifacts/portfolio/` (185 catalog rows, PCA, hybrid vs baseline CSVs) |
| **Lab 04** | Stable Diffusion v1.5 Controlled Image Synthesis | `.\.venv-lab04\Scripts\python.exe -m pytest labs/lab04_stable_diffusion/tests -q` | `$env:PORTFOLIO_MODE="full"; .\.venv-lab04\Scripts\python.exe -m labs.lab04_stable_diffusion.src.run` | Hugging Face cache (`runwayml/stable-diffusion-v1-5`, rev `451f4fe1...`) | CPU for `PORTFOLIO_MODE=replay`; NVIDIA GPU (4+ GB VRAM, FP16 + `model_cpu_offload`) for `full` | `labs/lab04_stable_diffusion/artifacts/portfolio/` (13 PNGs, `generations.csv`, intermediate snapshots) |
| **Lab 05** | Multimodal Visual Question Answering (LLaVA 0.5B) | `.\.venv-lab06\Scripts\python.exe -m pytest labs/lab05_multimodal/tests -q` | `$env:LAB05_MODE="LIVE"; .\.venv-lab06\Scripts\python.exe -m labs.lab05_multimodal.src.run` | Hugging Face cache (`llava-hf/llava-interleave-qwen-0.5b-hf`, rev `1090956d...`) | CPU for `LAB05_MODE=REPLAY`; NVIDIA GPU (4+ GB VRAM, FP16) for `LIVE` | `labs/lab05_multimodal/evidence/` (`verified_runs.json`, 17 unique runs / 19 rows, `token_budget.png`) |
| **Lab 06** | Autonomous Tool-Calling Agents (OpenAI Agents SDK) | `.\.venv-lab06\Scripts\python.exe -m pytest labs/lab06_agents/tests -q` | `.\.venv-lab06\Scripts\python.exe -m labs.lab06_agents.src.experiments` | LM Studio (`http://localhost:1234/v1` with `mistralai/ministral-3-3b`) | CPU for offline trace verification; 4+ GB VRAM for live agent loop | `labs/lab06_agents/outputs/runs/runs.jsonl`, `outputs/traces/*.json`, `outputs/metadata/*.json` |
| **Lab 07** | Local Engineering Knowledge Copilot (Capstone) | `.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests -q` | `.\labs\lab07_final_project\.venv\Scripts\python.exe labs/lab07_final_project/evaluation/run_benchmark.py` | Redis Stack (`localhost:6380`) for dense/hybrid retrieval; LM Studio (`localhost:1234/v1`) for live agent generation | CPU for retrieval & unit tests; 4+ GB VRAM for live LM Studio generation | `labs/lab07_final_project/evaluation/baselines/*.json` & `artifacts/portfolio/*.png` |

---

## 2. Separation of Offline Deterministic Verification vs. Live Integration

1. **Zero Undocumented Network Dependencies in CI**:
   - All GitHub Actions workflows (`.github/workflows/labs12-ci.yml`, `labs3-4-ci.yml`, `lab5-ci.yml`, `lab6-ci.yml`, `lab7-ci.yml`) execute strictly offline on `ubuntu-latest` CPU runners (with a local Redis Stack service container in `lab7-ci.yml`).
   - No CI workflow calls external commercial APIs or downloads unverified multi-gigabyte model checkpoints during standard unit tests.
2. **Explicit Provenance & Replay Integrity**:
   - Labs 3, 4, 5, 6, and 7 explicitly label historical replay runs (`HISTORICAL_REPLAY`, `VERIFIED_COMMITTED_EVIDENCE`) and never conflate artifact replay with fresh GPU/LLM execution.
   - Cryptographic SHA-256 digests validate datasets (`dataset_integrity_manifest.json`), generated images (`generations.csv`), and workspace reports (`manifest.json`).
   - When a live service (such as LM Studio) is offline, evaluation scripts (`eval_generation.py`, `run_live_acceptance.py`) record `BLOCKED` / `NOT_VERIFIED` rather than fabricating synthetic passes.
