# Project-Wide CUDA & Environment Baseline Audit

**Date:** 2026-10-30  
**Host Environment:** Windows 11 x64  
**Hardware GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM)  
**NVIDIA Driver:** 610.88, CUDA Driver: 13.3  
**Audit Trigger:** Notebook `labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb` was launching via `labs/lab03_rag/.venv/Scripts/python.exe` (CPU-only PyTorch `2.14.1+cpu`), causing GPU multimodal inference to fail.

---

## 1. Initial State Findings (Forensic Audit)

1. **Jupyter Kernels Misconfiguration:**
   - Only two kernels were registered in `%APPDATA%\jupyter\kernels`: `lab03-rag` and `lab05-cuda`.
   - `lab01`, `lab04`, `lab06`, and `lab07` kernels were completely absent from Jupyter.
   - All lab notebooks except Lab 07 had generic kernelspec metadata `{"name": "python3"}`.
   - Jupyter defaulted generic `python3` notebooks to whichever Python kernel was discovered first or recently active, resulting in `01_llava_vqa.ipynb` executing under `lab03_rag/.venv` (CPU-only `torch 2.14.1+cpu`).

2. **Lab 05 Environment Incompleteness:**
   - `labs/lab05_multimodal/.venv` contained verified CUDA baseline `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`, but dependencies from `requirements.txt` (`transformers`, `accelerate`, `safetensors`, `pandas`, `matplotlib`, `pytest`, `nbclient`) were missing.
   - In addition, PyCharm had injected a non-standard `jetTransient` property into cell outputs, failing `nbformat.validate()`.

3. **Background VRAM Occupancy:**
   - `nvidia-smi` revealed that an orphaned `llama-server.exe` (PID 10736) from prior agent/LM Studio sessions was holding 3405 MiB out of 4096 MiB VRAM.

4. **Multi-Environment Fragmentation:**
   - `.venv-lab04` had full CUDA PyTorch `2.14.1+cu130`, Diffusers `0.40.0`, Transformers `5.17.0`, and Accelerate `1.15.0`, passing real GPU compute tests.
   - `labs/lab07_final_project/.venv` had CUDA PyTorch `2.14.1+cu130`, passing real GPU compute tests.
   - `labs/lab03_rag/.venv` had CPU-only `torch 2.14.1+cpu` (CPU embeddings policy, correctly kept).
   - `.venv-lab06` had clean API-only packages without unnecessary PyTorch overhead.

---

## 2. Baseline Audit Matrix

| Lab | Notebook | Current Interpreter | Python | Torch | CUDA Build | GPU Available | Required Backend | Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | `01_spam_classification.ipynb`<br>`02_text_generation.ipynb` | `.venv-lab04\Scripts\python.exe` (unlinked `python3`) | 3.13.5 | 2.14.1+cu130 | 13.0 | True | CUDA | Register `lab01-transformer`, bind notebooks, install `nltk` into `.venv-lab04`. |
| **Lab 02** | *(CLI / pytest)* | `.venv-lab06\Scripts\python.exe` | 3.13.5 | None | None | False | CPU / WSL2 | Maintain CPU API client on Windows; keep vLLM isolated in WSL2 (`vllm-lab/`). |
| **Lab 03** | `01_redis_csv_rag.ipynb` | `labs\lab03_rag\.venv\Scripts\python.exe` | 3.13.5 | 2.14.1+cpu | None | False | CPU | Retain CPU embeddings; re-bind notebook strictly to `lab03-rag` to prevent cross-contamination. |
| **Lab 04** | `01_text_to_image.ipynb` | `.venv-lab04\Scripts\python.exe` (unlinked `python3`) | 3.13.5 | 2.14.1+cu130 | 13.0 | True | CUDA (RTX 3050, 4GB) | Register `lab04-cuda`, bind notebook to `lab04-cuda`, verify SD 1.5 GPU inference smoke. |
| **Lab 05** | `01_llava_vqa.ipynb` | `labs\lab03_rag\.venv\Scripts\python.exe` (MISLINKED) | 3.13.5 | 2.14.1+cpu | None | False | CUDA (RTX 3050, 4GB) | Fix interpreter to `labs\lab05_multimodal\.venv`, install missing deps without upgrading torch, fix `jetTransient`, run LIVE acceptance. |
| **Lab 06** | `01_openai_agents_sdk.ipynb` | `.venv-lab06\Scripts\python.exe` (unlinked `python3`) | 3.13.5 | None | None | False | CPU / LM Studio | Register `lab06-agents`, bind notebook, preserve local LM Studio server separation. |
| **Lab 07** | `01_final_project.ipynb` | `labs\lab07_final_project\.venv\Scripts\python.exe` (unregistered `lab07`) | 3.13.5 | 2.14.1+cu130 | 13.0 | True | CUDA (RTX 3050, 4GB) | Register `lab07` kernel in Jupyter, verify test suite and Redis integration. |

---

## 3. Immediate Action Plan

1. **Jupyter Kernel Normalization:** Register dedicated kernels for all labs with absolute interpreter paths.
2. **Notebook Kernelspec Synchronization:** Update every notebook's metadata so that Jupyter/PyCharm automatically picks the verified interpreter.
3. **Lab 05 Dependencies & Validation:** Install matching `transformers==5.17.0`, `accelerate==1.15.0`, etc. into `labs/lab05_multimodal/.venv`, strip invalid `jetTransient`, and run LIVE LLaVA inference.
4. **Lab 04 Smoke Verification:** Verify SD 1.5 pipeline loading and memory-offloaded GPU smoke inference within 4GB VRAM.
5. **Automation & Health Scripts:** Deliver `scripts/check_ml_environments.py`, `scripts/setup_jupyter_kernels.ps1`, and `scripts/verify_cuda.ps1`.
