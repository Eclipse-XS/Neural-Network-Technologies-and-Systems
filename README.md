# Neural Network Technologies and Systems

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for isolated Windows environments, pinned dependencies, model/data reconstruction and live smoke tests for Labs 1-7.

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

## Setup and prerequisites

Use Python 3.13 and the per-lab environment/lock, from the repository root:

```powershell
.\setup.ps1 -Lab 1
.\run.ps1 -Lab 1 -Smoke
.\run.ps1 -Lab 1
```

Use `-Cuda` for Labs 4-5. Labs 2, 3, 6 and 7 use LM Studio with a tool-capable model, CPU offload and at least 16384 context tokens. Lab 3 Redis uses port 6379; Lab 7 uses 6380. Lab 7 setup provisions its Redis image and embeddings; its launcher validates the external services before starting Streamlit. The complete source/checksum, service and mode instructions are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Do not combine all labs in a root virtual environment.

## Capstone Architecture (Lab 7)

[`labs/lab07_final_project/src/knowledge_copilot/`](labs/lab07_final_project/README.md) organizes the RAG copilot into layered modules:

- `domain/`: typed Pydantic/dataclass contracts (`Document`, `Chunk`, `EvidenceChunk`, `Turn`, `GroundingStatus`, `RetrievalMode`)
- `rag/`: structure-aware Markdown/PDF/text extractors & chunkers (`max_tokens=240`), `multilingual-e5-small` dense embeddings, `BM25Retriever`, and Reciprocal Rank Fusion (`reciprocal_rank_fusion`)
- `agents/`: ReAct orchestrator, bounded AST math (`safe_calculate`), citation/evidence verifier (`verify_grounding`), and empty/no-evidence recovery handlers
- `infrastructure/`: Redis Stack HNSW/FLAT vector store, SQLite WAL repository, and LM Studio OpenAI client
- `api/` & `ui/`: FastAPI service (`/api/v1/chat`, `/api/v1/documents`, `/api/v1/health/live`, `/api/v1/health/ready`, `/metrics`) and Streamlit web client
