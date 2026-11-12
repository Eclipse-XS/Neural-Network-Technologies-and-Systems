# Lab 07 — Local Engineering Knowledge Copilot

## Overview

End-to-end local-first agentic RAG application combining structure-aware Markdown/PDF/text ingestion, hybrid retrieval (**Redis Stack** dense `intfloat/multilingual-e5-small` cosine search + `BM25Retriever` via Reciprocal Rank Fusion), an OpenAI Agents SDK ReAct orchestrator over LM Studio, an AST arithmetic sandbox (`safe_calculate`), citation/grounding verification (`verify_grounding`), and **FastAPI** + **Streamlit** interfaces.

## Main Implementation Files

- Canonical modular monolith ([`src/knowledge_copilot/`](src/knowledge_copilot/)):
  - [`domain/`](src/knowledge_copilot/domain/) — `Document`, `Chunk`, `EvidenceChunk`, `Turn`, `GroundingStatus`, `RetrievalMode`
  - [`rag/`](src/knowledge_copilot/rag/) — `extractors.py`, `splitters.py` (structure-aware `max_tokens=240` chunking), `embeddings.py` (`multilingual-e5-small`), `lexical.py` (`BM25Retriever`), `hybrid.py` (`reciprocal_rank_fusion`)
  - [`agents/`](src/knowledge_copilot/agents/) — `orchestrator.py`, `prompts.py`, `tools.py` (`safe_calculate`), `verifier.py` (`verify_grounding`), `recovery.py`
  - [`application/`](src/knowledge_copilot/application/) — `ingestion_service.py`, `query_service.py`, `session_service.py`
  - [`infrastructure/`](src/knowledge_copilot/infrastructure/) — `redis_store.py` (port `6380`), `sqlite_repo.py` (WAL mode), `llm_client.py` (`http://localhost:1234/v1`)
  - [`api/`](src/knowledge_copilot/api/) — FastAPI `create_app()` / `app` (`routers/chat.py`, `routers/documents.py`, `routers/health.py`)
  - [`ui/`](src/knowledge_copilot/ui/streamlit_app.py) — Streamlit multi-session web UI
  - [`observability/`](src/knowledge_copilot/observability/) — JSON logging, Prometheus metrics, and tracing
- Notebook & Corpus Assets:
  - [`notebooks/01_final_project.ipynb`](notebooks/01_final_project.ipynb) — runnable walkthrough of ingestion, BM25 + RRF, AST math, SQLite session isolation, and `verify_grounding` (with optional `LAB07_RUN_LIVE=1`)
  - [`assets/knowledge/`](assets/knowledge/) — indexed Markdown lab reports and architecture/runbook/postmortem notes + `manifest.json`
- Configuration & Legacy Compatibility:
  - [`docker-compose.yml`](docker-compose.yml) (Redis Stack on `127.0.0.1:6380`), [`requirements.txt`](requirements.txt), [`setup.ps1`](setup.ps1), [`.env.example`](.env.example), [`src/`](src/) compatibility modules, [`web_app.py`](web_app.py), [`app.py`](app.py)

## How to Run

1. Install environment dependencies and start Redis Stack (`127.0.0.1:6380`):

   ```powershell
   .\labs\lab07_final_project\setup.ps1
   docker compose -f labs/lab07_final_project/docker-compose.yml up -d redis
   ```

2. Ensure LM Studio is running at `http://localhost:1234/v1` with a tool-capable model (e.g., `mistralai/ministral-3-3b`).

3. Start the FastAPI server and Streamlit UI from the repository root:

   ```powershell
   .\labs\lab07_final_project\.venv\Scripts\python.exe -m uvicorn labs.lab07_final_project.src.knowledge_copilot.api.app:app --host 127.0.0.1 --port 8000
   .\labs\lab07_final_project\.venv\Scripts\streamlit.exe run labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py --server.port 8501
   ```

4. Key REST endpoints:
   - `GET /api/v1/health/live`, `GET /api/v1/health/ready`, `GET /metrics`
   - `GET /api/v1/documents`, `POST /api/v1/documents`, `DELETE /api/v1/documents/{document_id}`
   - `POST /api/v1/chat` (`{"query": "...", "session_id": "...", "retrieval_mode": "hybrid"}`)

## Representative Example

Running [`notebooks/01_final_project.ipynb`](notebooks/01_final_project.ipynb) executes the live runtime modules over [`assets/knowledge/`](assets/knowledge/):
- Extracts and splits the 8 indexed Markdown knowledge documents into structure-aware chunks (`max_tokens=220`).
- Indexes chunks into `BM25Retriever` and merges lexical + dense candidates with `reciprocal_rank_fusion`.
- Verifies safe AST math (`safe_calculate("17 - 13") == 4`) and blocks code injection (`__import__('os')...`).
- Validates citations with `verify_grounding` (`GROUNDED` for retrieved `[<doc_id>#<chunk_id>]`, rejecting fabricated citations such as `[nonexistent_doc#c_99]`).
