# Gemini Recovery Baseline & Forensic Audit

Date: 2026-10-30. Context: Continuation of unfinished Codex session for Labs 5–7 with Lab 7 final acceptance as primary target.

## 1. Repository and Git Forensic State

| Property | Value |
|---|---|
| Repository Root | `C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems` |
| Active Branch | `codex/labs-3-4-portfolio-upgrade` |
| HEAD Commit | `01ec07f9ce7394365b22e0308fe38e01c30790a8` |
| Staged / Committed by Gemini | None (Rule 8 strictly observed: no push, merge, or unauthorized commit) |
| Pre-existing User Work Preserved | `.gitignore`, `NNTS_VARIANT_2/`, `scripts/fix_lab1_report_layout.py`, `vllm-lab/` |
| Working Tree Backup | Preserved in `labs/lab07_final_project/outputs/baseline_backup/` (`initial_git_status.txt`, `initial_git_diff.patch`, `initial_head.txt`) |

### Modified Files in Working Tree (from Codex):
- `README.md`
- `labs/lab06_agents/assets/workspace/lab03_report.md`
- `labs/lab06_agents/assets/workspace/lab04_report.md`
- `labs/lab06_agents/assets/workspace/manifest.json`
- `labs/lab07_final_project/README.md`
- `labs/lab07_final_project/assets/knowledge/lab03_rag_report.md`
- `labs/lab07_final_project/assets/knowledge/lab04_stable_diffusion_report.md`
- `labs/lab07_final_project/assets/knowledge/manifest.json`
- `labs/lab07_final_project/experiments/retrieval_queries.json`
- `labs/lab07_final_project/experiments/scenarios.json`
- `labs/lab07_final_project/requirements.txt`
- `labs/lab07_final_project/setup.ps1`
- `labs/lab07_final_project/src/agent.py`
- `labs/lab07_final_project/src/config.py`
- `labs/lab07_final_project/src/experiments.py`
- `labs/lab07_final_project/src/knowledge_base.py`
- `labs/lab07_final_project/src/tools.py`
- `labs/lab07_final_project/tests/test_knowledge_base.py`
- `labs/lab07_final_project/tests/test_scenarios.py`
- `labs/lab07_final_project/tests/test_tools.py`
- `pyproject.toml`

### Untracked Files Created by Codex:
- `.github/workflows/labs-6-7-ci.yml`
- `docs/audits/FINAL_PORTFOLIO_ACCEPTANCE.md`
- `labs/lab07_final_project/artifacts/` (and subdirectories)
- `labs/lab07_final_project/requirements-ci.txt`
- `labs/lab07_final_project/web_app.py`

## 2. Infrastructure Verification

| Service / Dependency | Version / Identity | Status |
|---|---|---|
| Python Runtime | Python 3.13.5 (64-bit AMD64) | PASS (`labs/lab07_final_project/.venv/Scripts/python.exe`) |
| PyTorch / CUDA | `torch 2.14.1+cu130`, CUDA 13.0, Device: `NVIDIA GeForce RTX 3050 Laptop GPU` | PASS |
| Redis Server | `8.10.3` running on `localhost:6380` (WSL2), modules: `['search', 'vectorset']` | PASS |
| LM Studio | OpenAI-compatible server at `http://localhost:1234/v1`, active model: `mistralai/ministral-3-3b` | PASS |
| Docker Desktop | Broken Windows socket (`sailor-ingest.sock`); NOT required since WSL Redis is active | BYPASS (Working as designed without Docker) |
| Port 8000 | Unrelated Uvicorn process; strictly untouched | PRESERVED |

## 3. Forensic Analysis of Interrupted Work

1. **Last Complete Live Agent Batch**:
   - Batch ID: `experiment_146e11345e50`
   - Timestamp: `2026-10-30 06:26:46`
   - Runs: 16 runs executed (`runs.jsonl`, 302 KB)
   - Traces: Recorded in `outputs/traces/`
   - Metadata: `index.json`, `embedding.json`, `chunks.json`, `environment.json`
2. **Current Retrieval Configuration**:
   - Embedding Model: `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, 384 dim, normalized)
   - Chunker: 112 chunks, max tokens 239 <= 240 token budget, 0 truncated chunks.
   - Retrieval mode: `hybrid` (dense + lexical BM25 with Reciprocal Rank Fusion, weights 2.0 dense / 1.0 lexical).
   - Evidence quality tagging and exclusion of quote contamination.
3. **Interrupted Work Identified**:
   - Codex was executing Streamlit functional E2E testing against the live backend via `streamlit.testing.v1.AppTest`.
   - The test harness crashed trying to read `ChatMessage.value`, which does not exist on `ChatMessage` (`ChatMessage` exposes child elements like `.markdown[0].value`).
   - The UI test was left incomplete.
   - The final independent agent evaluation and report comparing baseline vs final were incomplete.
   - Academic matrix in `FINAL_PORTFOLIO_ACCEPTANCE.md` was still in an intermediate offline state stating that live services were blocked, even though live services subsequently became available.

## 4. Immediate Acceptance Plan

1. **Phase B & C**: Re-run and verify RAG ingestion and retrieval on the live Redis backend. Create a fresh held-out evaluation query set to verify unbiased retrieval generalization.
2. **Phase D**: Inspect and audit batch `experiment_146e11345e50` traces. Analyze citations, arithmetic policy enforcement vs autonomous calls, and memory isolation.
3. **Phase E**: Complete the Streamlit E2E test by writing a robust test harness in `tests/test_web_app.py` properly accessing `markdown[0].value`, testing real live question answering, session resume, and error handling.
4. **Phase F**: Run frozen held-out agent evaluation with distinct run IDs and evaluate correctness, groundedness, and latency.
5. **Phase G**: Run all test suites across Labs 5, 6, and 7, test replay notebook, verify CI workflows.
6. **Phase H & I**: Update academic compliance matrix and portfolio documentation.
7. **Phase J**: Deliver `GEMINI_FINAL_LIVE_ACCEPTANCE.md` and `WINDOWS_RUNBOOK.md`.
