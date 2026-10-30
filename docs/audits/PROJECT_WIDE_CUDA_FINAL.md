# Project-Wide CUDA, PyTorch, and Jupyter Environments Overhaul: Final Acceptance Report

**Date:** 2026-10-30  
**Host Environment:** Windows 11 Home x64  
**Physical Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM)  
**NVIDIA Driver:** 610.88, CUDA Driver: 13.3  
**Status:** PASS (All environments, kernels, notebooks, and GPU compute tests verified)

---

## 1. Початкові проблеми (Initial Problems Identified)

1. **Jupyter Kernels Misbinding & Cross-Contamination:**
   - Notebook `labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb` was executing under `labs/lab03_rag/.venv/Scripts/python.exe`, which contained CPU-only PyTorch (`2.14.1+cpu`), causing multimodal GPU inference to fail.
   - Notebooks across Labs 1, 3, 4, 5, 6 had generic kernelspec `"name": "python3"` in their metadata. Jupyter resolved `"python3"` arbitrarily to the first registered or active kernel.
   - Only two kernels (`lab03-rag` and `lab05-cuda`) were registered in `%APPDATA%\jupyter\kernels`. Kernels for Lab 1, Lab 4, Lab 6, and Lab 7 did not exist in Jupyter.
2. **Missing Dependencies in Lab 05:**
   - `labs/lab05_multimodal/.venv` had the verified CUDA baseline `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`, but `transformers`, `accelerate`, `safetensors`, `pandas`, `matplotlib`, and `pytest` were not installed.
   - `jetTransient` proprietary metadata from PyCharm was present in cell outputs, causing `nbformat.validate()` to raise `NotebookValidationError`.
3. **Lab 01 Missing Dependencies:**
   - Lab 01 tests failed due to missing `nltk` in the shared `.venv-lab04` environment.
4. **VRAM Constraints & Background Processes:**
   - An orphaned background process `llama-server.exe` (PID 10736) held ~3.4 GB of the 4 GB VRAM from previous LM Studio runs, creating an imminent Risk of CUDA OOM for heavy models.

---

## 2. Остаточна архітектура середовищ (Environment Architecture)

Rather than duplicating PyTorch installations across 7+ environments, the architecture consolidates compatible stacks while maintaining clean boundaries for CPU-only and Linux/WSL2 components:

1. **Shared CUDA Transformer & Diffusion Stack (`.venv-lab04`):**
   - Serves **Lab 01** (Transformer Encoder/Decoder) and **Lab 04** (Stable Diffusion 1.5).
   - Python: 3.13.5
   - PyTorch: `2.14.1+cu130`, CUDA 13.0
   - Stack: `diffusers 0.40.0`, `transformers 5.17.0`, `accelerate 1.15.0`, `nltk 3.10.4`, `scikit-learn 1.7.2`, `ipykernel 7.3.0`.
   - Kernels registered: `lab01-transformer`, `lab04-cuda`.

2. **Dedicated Multimodal LLaVA CUDA Stack (`labs/lab05_multimodal/.venv`):**
   - Serves **Lab 05** (LLaVA-Interleave Qwen 0.5B).
   - Preserved verified baseline: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, CUDA 12.8.
   - Fully resolved dependencies: `transformers 5.17.0`, `accelerate 1.15.0`, `safetensors 0.8.0`, `pillow 12.3.0`, `pandas 3.0.6`, `matplotlib 3.11.2`, `pytest 9.1.1`, `ipykernel 7.3.0`.
   - Kernel registered: `lab05-cuda`.

3. **Dedicated Final Project CUDA Stack (`labs/lab07_final_project/.venv`):**
   - Serves **Lab 07** (Agentic RAG Final Project & Streamlit showcase).
   - Python: 3.13.5
   - PyTorch: `2.14.1+cu130`, CUDA 13.0
   - Stack: `redis 8.1.0`, `redisvl 0.27.2`, `sentence-transformers 6.1.0`, `langchain-huggingface 1.2.2`, `langchain-redis 0.2.6`, `openai-agents 0.22.3`, `streamlit 1.42.0`, `ipykernel 7.3.0`.
   - Kernels registered: `lab07`, `lab07-cuda`.

4. **CPU-Only RAG Stack (`labs/lab03_rag/.venv`):**
   - Serves **Lab 03** (Redis CSV RAG).
   - Python: 3.13.5
   - PyTorch: `2.14.1+cpu` (CPU embeddings policy strictly maintained per requirements).
   - Stack: `langchain-community 0.4.2`, `redis 8.1.0`, `redisvl 0.27.2`, `sentence-transformers 6.1.0`, `ipykernel 6.30.1`.
   - Kernel registered: `lab03-rag`.

5. **CPU / Agents API Client Stack (`.venv-lab06`):**
   - Serves **Lab 06** (OpenAI Agents SDK) and **Lab 02 Windows client**.
   - Python: 3.13.5
   - PyTorch: None (Zero torch bloat).
   - Stack: `openai-agents 0.22.3`, `openai 3.17.0`, `pydantic 2.13.5`, `httpx2 2.13.0`, `ipykernel 7.3.0`.
   - Kernel registered: `lab06-agents`.

6. **Isolated Linux/vLLM Runtime (`vllm-lab/`):**
   - Serves **Lab 02 vLLM backend** in WSL2 / Ubuntu, completely decoupled from Windows Python environments.

---

## 3. Таблиця відповідності лабораторних та їхніх Kernels

| Lab | Notebook | Registered Kernel | Kernel Display Name | Interpreter Path | Python | Torch | CUDA Build | GPU Compute | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | `01_spam_classification.ipynb`<br>`02_text_generation.ipynb` | `lab01-transformer` | Lab 01 Transformer (CUDA RTX 3050) | `.venv-lab04\Scripts\python.exe` | 3.13.5 | 2.14.1+cu130 | 13.0 | PASS | PASS |
| **Lab 02** | *(CLI / pytest)* | `lab06-agents` | Lab 06 OpenAI Agents SDK | `.venv-lab06\Scripts\python.exe` | 3.13.5 | None | None | N/A (CPU) | PASS |
| **Lab 03** | `01_redis_csv_rag.ipynb` | `lab03-rag` | Lab 03 Redis CSV RAG (CPU) | `labs\lab03_rag\.venv\Scripts\python.exe` | 3.13.5 | 2.14.1+cpu | None | N/A (CPU) | PASS |
| **Lab 04** | `01_text_to_image.ipynb` | `lab04-cuda` | Lab 04 CUDA RTX 3050 (Stable Diffusion) | `.venv-lab04\Scripts\python.exe` | 3.13.5 | 2.14.1+cu130 | 13.0 | PASS | PASS |
| **Lab 05** | `01_llava_vqa.ipynb` | `lab05-cuda` | Lab 05 CUDA RTX 3050 (LLaVA) | `labs\lab05_multimodal\.venv\Scripts\python.exe` | 3.13.5 | 2.11.0+cu128 | 12.8 | PASS | PASS |
| **Lab 06** | `01_openai_agents_sdk.ipynb` | `lab06-agents` | Lab 06 OpenAI Agents SDK | `.venv-lab06\Scripts\python.exe` | 3.13.5 | None | None | N/A (CPU) | PASS |
| **Lab 07** | `01_final_project.ipynb` | `lab07` | Lab 07 Final Project Agentic RAG (CUDA) | `labs\lab07_final_project\.venv\Scripts\python.exe` | 3.13.5 | 2.14.1+cu130 | 13.0 | PASS | PASS |

---

## 4. Версії Python, PyTorch, CUDA

| Component | Version | Verification Check |
| :--- | :--- | :--- |
| **Host Python** | 3.13.5 (CPython 64-bit) | Standardized across Windows venvs |
| **PyTorch (Lab 05 Multimodal)** | `2.11.0+cu128` | Verified baseline intact; no reinstallation |
| **Torchvision (Lab 05)** | `0.26.0+cu128` | Pre-installed matching wheel |
| **PyTorch (Lab 01, 04, 07)** | `2.14.1+cu130` | Production CUDA build |
| **PyTorch (Lab 03 RAG)** | `2.14.1+cpu` | CPU-only embeddings preserved |
| **CUDA Driver / UMD** | 13.3 (Driver 610.88) | Native Windows NVIDIA driver |
| **Transformers** | 5.17.0 | Pinned and compatible across Labs 4, 5, 7 |
| **Diffusers** | 0.40.0 | Pinned for Lab 04 SD 1.5 |
| **Accelerate** | 1.15.0 | Pinned for memory-offload execution |

---

## 5. Результати GPU Compute Tests

Automated verification script `scripts/verify_cuda.ps1` executed real 1024x1024 matrix multiplication on `cuda:0` across all CUDA environments:

```text
=== CUDA & GPU Hardware Verification ===
Repository root: C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems

--- nvidia-smi Check ---
NVIDIA GeForce RTX 3050 Laptop GPU, 610.88, 4096 MiB, 2758 MiB, 1205 MiB

--- CUDA Environments Verification ---

Testing: Lab 01 & Lab 04 (Transformer / Stable Diffusion)
Interpreter: .venv-lab04\Scripts\python.exe
  Lab 01 & Lab 04 | Kernel lab01-transformer / lab04-cuda | 2.14.1+cu130 | NVIDIA GeForce RTX 3050 Laptop GPU | GPU COMPUTE PASS (alloc=20.12MB,res=42.0MB)

Testing: Lab 05 (Multimodal LLaVA)
Interpreter: labs\lab05_multimodal\.venv\Scripts\python.exe
  Lab 05 (Multimodal LLaVA) | Kernel lab05-cuda | 2.11.0+cu128 | NVIDIA GeForce RTX 3050 Laptop GPU | GPU COMPUTE PASS (alloc=20.12MB,res=42.0MB)

Testing: Lab 07 (Agentic RAG Final Project)
Interpreter: labs\lab07_final_project\.venv\Scripts\python.exe
  Lab 07 (Agentic RAG Final Project) | Kernel lab07 / lab07-cuda | 2.14.1+cu130 | NVIDIA GeForce RTX 3050 Laptop GPU | GPU COMPUTE PASS (alloc=20.12MB,res=42.0MB)

----------------------------------------
ALL CUDA ENVIRONMENTS PASSED REAL GPU COMPUTE TESTS!
```

---

## 6. Результати LLaVA LIVE Inference (Lab 05 Acceptance)

1. **Model:** `llava-hf/llava-interleave-qwen-0.5b-hf` (revision `1090956dd1c79bc93ae98dcf395590369435ec91`)
2. **Device Strategy:** `cuda:0` with `torch.float16` and SDPA attention.
3. **VRAM Footprint:**
   - Model weights in VRAM: **1678.65 MB** (~1.68 GB)
   - Peak allocation during inference: **< 2.0 GB** (Leaves > 2 GB VRAM free on RTX 3050).
4. **Live Single Smoke Output:**
   - Question: *"Is the cabin on the left, in the center, or on the right side of the image?"*
   - Generated Answer: **`left`**
   - Tokens Generated: 2, Ended with EOS: True.
5. **Notebook Full Execution:**
   - `labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb` executed end-to-end via `lab05-cuda` Jupyter kernel with `NotebookClient`.
   - Cell 8 printed verified provenance:
     ```text
     Python: C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems\labs\lab05_multimodal\.venv\Scripts\python.exe
     PyTorch: 2.11.0+cu128
     CUDA: True
     GPU: NVIDIA GeForce RTX 3050 Laptop GPU
     ```
   - All 9 unit tests in `labs/lab05_multimodal/tests` pass with zero failures:
     `9 passed in 0.41s`

---

## 7. Результати Stable Diffusion Smoke (Lab 04 Acceptance)

1. **Model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
2. **Device Strategy:** `model_cpu_offload` with `torch.float16`.
3. **Execution Verification:**
   - Component weights loaded onto GPU sequentially via Accelerate hooks.
   - 1-step test generation executed on `cuda:0` in 8.36 seconds.
   - Result: Generated image of size `(512, 512)` without exceeding 4 GB VRAM limit.
   - All 24 tests in `labs/lab04_stable_diffusion/tests` pass with zero errors (`24 passed in 3.01s`).

---

## 8. Стан vLLM / WSL2 (Lab 02)

1. Lab 02 tests run against OpenAI-compatible endpoints.
2. The Windows client runtime is tested via `.venv-lab06` (`15 passed in 1.52s`).
3. The vLLM server configuration is isolated under `vllm-lab/` for execution under WSL2 Ubuntu without contaminating Windows wheels.

---

## 9. Стан Lab 07 (Agentic RAG Final Project)

1. Tests executed with `labs/lab07_final_project/.venv/Scripts/python.exe`.
2. Multilingual E5 embeddings run efficiently on the RTX 3050 GPU.
3. Redis integration, Hybrid Retrieval, Agent tools, and Scenarios:
   `32 passed, 3 skipped in 36.87s`.
4. Dedicated Jupyter Kernel `lab07` registered and bound to `01_final_project.ipynb`.

---

## 10. Повний статус тестів по всіх лабораторних

| Test Suite | Python Interpreter | Passed | Skipped | Failed | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 01 Tests** (`labs/lab01_transformer/tests`) | `.venv-lab04` | 71 | 0 | 0 | **PASS** |
| **Lab 02 Tests** (`labs/lab02_llm_api/tests`) | `.venv-lab06` | 15 | 0 | 0 | **PASS** |
| **Lab 03 Tests** (`labs/lab03_rag/tests`) | `labs/lab03_rag/.venv` | 41 | 0 | 0 | **PASS** |
| **Lab 04 Tests** (`labs/lab04_stable_diffusion/tests`) | `.venv-lab04` | 24 | 0 | 0 | **PASS** |
| **Lab 05 Tests** (`labs/lab05_multimodal/tests`) | `labs/lab05_multimodal/.venv` | 9 | 0 | 0 | **PASS** |
| **Lab 06 Tests** (`labs/lab06_agents/tests`) | `.venv-lab06` | 33 | 1 | 0 | **PASS** |
| **Lab 07 Tests** (`labs/lab07_final_project/tests`) | `labs/lab07_final_project/.venv` | 32 | 3 | 0 | **PASS** |
| **Total Test Count** | — | **225** | **4** | **0** | **ALL PASS** |

---

## 11. Список створених та змінених файлів

### Нові автоматизовані скрипти:
- `scripts/check_ml_environments.py` — комплексний аудит інтерпретаторів, ядер, метаданих ноутбуків і CUDA обчислень.
- `scripts/setup_jupyter_kernels.ps1` — ідемпотентна реєстрація ядер у Jupyter та нормалізація kernelspec.
- `scripts/verify_cuda.ps1` — перевірка `nvidia-smi` та реальне GPU тензорне обчислення у всіх CUDA середовищах.

### Нові аудит-звіти:
- `docs/audits/PROJECT_WIDE_CUDA_BASELINE.md` — вихідний forensic аудит проблем і матриця зв'язків.
- `docs/audits/PROJECT_WIDE_CUDA_FINAL.md` — остаточний звіт та документація прийняття.

### Оновлені файли конфігурації та коду:
- `README.md` — додано розділ GPU Environment Setup та інструкції запуску.
- `labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb` — очищено від `jetTransient`, прив'язано до `lab05-cuda`, виконано повний запуск через Jupyter kernel.
- `labs/lab01_transformer/notebooks/*.ipynb` — прив'язано kernelspec до `lab01-transformer`.
- `labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb` — прив'язано kernelspec до `lab03-rag`.
- `labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb` — прив'язано kernelspec до `lab04-cuda`.
- `labs/lab06_agents/notebooks/01_openai_agents_sdk.ipynb` — прив'язано kernelspec до `lab06-agents`.
- `labs/lab07_final_project/notebooks/01_final_project.ipynb` — перевірено та прив'язано до зареєстрованого ядра `lab07`.

---

## 12. Невирішені обмеження (Known Technical Limitations)

1. **4 GB VRAM Budget:**
   - Laptop RTX 3050 has 4096 MiB physical VRAM.
   - Do NOT run LM Studio / `llama-server.exe` simultaneously with Stable Diffusion or LLaVA, as `llama-server.exe` can occupy ~3.4 GB VRAM and trigger CUDA OOM.
   - When switching between Lab 6 (LM Studio) and Lab 4/5 (Stable Diffusion / LLaVA), ensure previous inference servers are stopped or unloaded from GPU.
2. **vLLM Linux Requirement:**
   - vLLM requires a Linux kernel. On Windows, it must run via WSL2 (see `vllm-lab/SETUP_RECORD.md`).

---

## 13. Інструкція запуску кожної GPU-лабораторної

### Швидка перевірка середовищ:
```powershell
# 1. Повна діагностика ядер та середовищ
& ".venv-lab04\Scripts\python.exe" scripts/check_ml_environments.py

# 2. Перевірка реального CUDA тензорного обчислення на GPU
powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1

# 3. Перереєстрація ядер (якщо налаштовується нова машина)
powershell -ExecutionPolicy Bypass -File scripts/setup_jupyter_kernels.ps1
```

### Запуск Lab 01 (Transformer):
У Jupyter або IDE відкрийте `labs/lab01_transformer/notebooks/01_spam_classification.ipynb` і оберіть ядро:
**`Lab 01 Transformer (CUDA RTX 3050)`** (`lab01-transformer`).

### Запуск Lab 04 (Stable Diffusion):
У Jupyter або IDE відкрийте `labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb` і оберіть ядро:
**`Lab 04 CUDA RTX 3050 (Stable Diffusion)`** (`lab04-cuda`).

### Запуск Lab 05 (LLaVA Multimodal):
У Jupyter або IDE відкрийте `labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb` і оберіть ядро:
**`Lab 05 CUDA RTX 3050 (LLaVA)`** (`lab05-cuda`).
Всі клітинки автоматично запускаються через `labs/lab05_multimodal/.venv/Scripts/python.exe` на NVIDIA GeForce RTX 3050.

### Запуск Lab 07 (Agentic RAG Final Project):
У Jupyter або IDE відкрийте `labs/lab07_final_project/notebooks/01_final_project.ipynb` і оберіть ядро:
**`Lab 07 Final Project Agentic RAG (CUDA)`** (`lab07`).
Для веб-додатку Streamlit:
```powershell
& "labs/lab07_final_project/.venv/Scripts/python.exe" -m streamlit run labs/lab07_final_project/web_app.py
```
