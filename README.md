# Neural Network Technologies and Systems

Code-first portfolio of seven modular deep learning, retrieval-augmented generation (RAG), multimodal VQA, and agentic AI laboratories implemented in Python and PyTorch.

## Laboratory Directory

| Lab | Title | Core Stack | Main Entrypoints |
|---|---|---|---|
| **01** | [Transformer from Scratch](labs/lab01_transformer/README.md) | PyTorch `TransformerEncoder` & `TransformerDecoder`, sinusoidal positional encoding, regex tokenizer | [01_spam_classification.ipynb](labs/lab01_transformer/notebooks/01_spam_classification.ipynb), [02_text_generation.ipynb](labs/lab01_transformer/notebooks/02_text_generation.ipynb) |
| **02** | [Local LLM API Client](labs/lab02_llm_api/README.md) | OpenAI Python SDK against LM Studio (`http://localhost:1234/v1`) | [`labs.lab02_llm_api.src.cli`](labs/lab02_llm_api/src/cli.py), [`client.py`](labs/lab02_llm_api/src/client.py) |
| **03** | [Redis Vector RAG](labs/lab03_rag/README.md) | LangChain, `all-MiniLM-L6-v2`, Redis Stack, hybrid catalog/semantic router | [01_redis_csv_rag.ipynb](labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb), [`rag.py`](labs/lab03_rag/src/rag.py) |
| **04** | [Stable Diffusion v1.5](labs/lab04_stable_diffusion/README.md) | Diffusers FP16, `model_cpu_offload`, PNDM scheduler, VAE/UNet latent inspection | [01_text_to_image.ipynb](labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb), [`run.py`](labs/lab04_stable_diffusion/src/run.py) |
| **05** | [Multimodal LLaVA VQA](labs/lab05_multimodal/README.md) | `llava-hf/llava-interleave-qwen-0.5b-hf` (SigLIP + MLP + Qwen2), CPU replay & CUDA inference | [01_llava_vqa.ipynb](labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb), [02_llava_live_inference.ipynb](labs/lab05_multimodal/notebooks/02_llava_live_inference.ipynb) |
| **06** | [Agentic Function Calling](labs/lab06_agents/README.md) | OpenAI Agents SDK, AST-safe calculator, bounded workspace tools, `SQLiteSession` | [01_openai_agents_sdk.ipynb](labs/lab06_agents/notebooks/01_openai_agents_sdk.ipynb), [`experiments.py`](labs/lab06_agents/src/experiments.py) |
| **07** | [Local Engineering Knowledge Copilot](labs/lab07_final_project/README.md) | Hybrid RAG (Redis Stack + `intfloat/multilingual-e5-small` + BM25 RRF), Grounding Verifier, FastAPI, Streamlit | [`app.py`](labs/lab07_final_project/src/knowledge_copilot/api/app.py), [`streamlit_app.py`](labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py), [01_final_project.ipynb](labs/lab07_final_project/notebooks/01_final_project.ipynb) |

## Repository Layout

```text
.
├── pyproject.toml
├── .gitignore
└── labs/
    ├── lab01_transformer/
    ├── lab02_llm_api/
    ├── lab03_rag/
    ├── lab04_stable_diffusion/
    ├── lab05_multimodal/
    ├── lab06_agents/
    └── lab07_final_project/
```

## Setup & Prerequisites

1. **Python >= 3.11** (see `pyproject.toml`). Each lab pins its own dependencies in `labs/<lab>/requirements.txt`:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r labs/lab07_final_project/requirements.txt
   ```

2. **Redis Stack** for vector storage (port `6379` in Lab 3, port `6380` in Lab 7):

   ```powershell
   docker compose -f labs/lab03_rag/docker-compose.yml up -d
   docker compose -f labs/lab07_final_project/docker-compose.yml up -d
   ```

3. **LM Studio Local API** for local LLM labs (Labs 2, 3, 6, 7): start LM Studio's OpenAI-compatible server on `http://localhost:1234/v1` with a tool-capable model such as `mistralai/ministral-3-3b`.

4. **CUDA GPU (optional for live runs)**: Labs 4 and 5 use PyTorch CUDA FP16 inference (`PORTFOLIO_MODE=live` / `LAB05_MODE=LIVE`) and support offline CPU replay modes (`PORTFOLIO_MODE=replay` / `LAB05_MODE=REPLAY`) over committed example outputs.

## Capstone Architecture (Lab 7)

[`labs/lab07_final_project/src/knowledge_copilot/`](labs/lab07_final_project/README.md) organizes the RAG copilot into layered modules:

- `domain/`: typed Pydantic/dataclass contracts (`Document`, `Chunk`, `EvidenceChunk`, `Turn`, `GroundingStatus`, `RetrievalMode`)
- `rag/`: structure-aware Markdown/PDF/text extractors & chunkers (`max_tokens=240`), `multilingual-e5-small` dense embeddings, `BM25Retriever`, and Reciprocal Rank Fusion (`reciprocal_rank_fusion`)
- `agents/`: ReAct orchestrator, bounded AST math (`safe_calculate`), citation/evidence verifier (`verify_grounding`), and empty/no-evidence recovery handlers
- `infrastructure/`: Redis Stack HNSW/FLAT vector store, SQLite WAL repository, and LM Studio OpenAI client
- `api/` & `ui/`: FastAPI service (`/api/v1/chat`, `/api/v1/documents`, `/api/v1/health/live`, `/api/v1/health/ready`, `/metrics`) and Streamlit web client
