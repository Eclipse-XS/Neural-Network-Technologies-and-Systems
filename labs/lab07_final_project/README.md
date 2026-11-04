# Local Engineering Knowledge Copilot

[![CI: Labs 6 & 7 Offline Suite](https://github.com/Eclipse-XS/Neural-Network-Technologies-and-Systems/actions/workflows/labs-6-7-ci.yml/badge.svg)](https://github.com/Eclipse-XS/Neural-Network-Technologies-and-Systems/actions/workflows/labs-6-7-ci.yml)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI: 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit: 1.42+](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![VectorDB: Redis Stack](https://img.shields.io/badge/VectorDB-Redis%20Stack%20(Port%206380)-DC382D.svg)](https://redis.io)
[![Inference: Local LM Studio](https://img.shields.io/badge/Inference-LM%20Studio%20(Local%20Port%201234)-8A2BE2.svg)](https://lmstudio.ai)

An enterprise-grade, privacy-oriented **Local Engineering Knowledge Copilot** designed to ingest complex technical documentation, perform hybrid dense-lexical retrieval, reason over multi-document evidence, execute safe AST-sandboxed calculations, and persist isolated multi-turn conversations without external cloud dependencies.

---

## 1. Product Capabilities and Supported Use Cases

The Copilot is engineered as a modular monolith adhering to strict local execution boundaries.

| Use Case ID | Name | Description | Key Guarantees |
|---|---|---|---|
| **UC-01** | **Document Ingestion & Lifecycle** | Ingest Markdown, plain text, and text-based PDFs into dual SQLite/Redis storage. | Strict 10MB limit; explicit rejection of scanned/empty PDFs requiring OCR (`ExtractionError`); idempotent uploads. |
| **UC-02** | **Document-Grounded Q&A** | Answer user queries grounded strictly in indexed documentation. | Returns synthesized answer, cited passages, exact chunk IDs, and source document filenames. |
| **UC-03** | **Cross-Document Comparison** | Contrast findings, metrics, and parameters across multiple disparate technical reports. | Multi-target cross-comparison routing; attributes claims to individual source documents. |
| **UC-04** | **Numerical & Arithmetic Analysis** | Answer quantitative queries requiring calculation over retrieved facts. | AST-sandboxed arithmetic evaluator (`calculate`); operands traceable to retrieved passages; deterministic recovery for missed tool calls. |
| **UC-05** | **Unsupported Information Refusal** | Abstain from hallucinating answers when evidence is absent from the corpus. | Strict negative control filtering; outputs explicit refusal and flags confidence as `insufficient_evidence`. |
| **UC-06** | **Session Memory & Isolation** | Maintain multi-turn context across chat sessions. | SQLite WAL-backed conversation store; session isolation; graceful degradation on transaction errors. |
| **UC-07** | **Knowledge Management** | Manage indexed documents (list, inspect status, delete, reindex). | Real-time status reporting (`processing`, `indexed`, `failed`); atomic cascading delete across Redis and SQLite. |
| **UC-08** | **Scientific Evaluation** | Run frozen evaluation benchmarks to measure retrieval quality and agent fidelity. | 100-query frozen benchmark; automated reporting of Recall@1/3/5, Hit@5, MRR, and latency. |
| **UC-09** | **System Diagnostics & Telemetry** | Inspect application health, dependencies, and operational metrics. | Structured JSON logs; Prometheus `/metrics` endpoint; `/api/v1/health/live` and `/ready` probes. |

---

## 2. System Architecture

The Copilot is built as a **modular monolith** with clean architectural layers:

```mermaid
flowchart TD
    User["Engineering User / Web Client"] -->|HTTP / UI| Streamlit["Streamlit UI (Port 8501)"]
    User -->|REST API Requests| FastAPI["FastAPI Server (Port 8000)"]
    Streamlit -->|HTTP Client / Services| FastAPI

    subgraph MonolithCore ["Knowledge Copilot Core"]
        FastAPI --> IngestSvc["Application: IngestionService"]
        FastAPI --> QuerySvc["Application: QueryService"]
        FastAPI --> SessionSvc["Application: SessionService"]
        
        QuerySvc --> Agent["Agent: AgentOrchestrator (ReAct <= 10 turns)"]
        Agent --> CalcTool["Tool: AST Calculator (Safe Math)"]
        Agent --> Verifier["Evidence Verifier & Citations Gate"]
        
        QuerySvc --> HybridEngine["RAG: Hybrid Engine (BM25 + Dense)"]
        IngestSvc --> HybridEngine
    end

    subgraph StorageAdapters ["Persistence & Indexing Adapters"]
        HybridEngine -->|Dense Vectors & Filtering| RedisDB["Redis Stack (Port 6380)"]
        IngestSvc -->|Vector Indexing / Deletion| RedisDB
        SessionSvc -->|Sessions, Turns & Metadata| SQLiteDB["SQLite Database (assistant_memory.sqlite)"]
        IngestSvc -->|Document Chunks & Hashes| SQLiteDB
    end

    subgraph LocalInference ["Local Model Serving"]
        Agent -->|OpenAI-Compatible REST (Port 1234)| LMStudio["LM Studio (Mistral / Qwen GGUF)"]
    end

    subgraph Observability ["Telemetry & Diagnostics"]
        FastAPI --> Prometheus["Prometheus Metrics (/metrics)"]
        FastAPI --> StructLog["Structured Tracing (X-Request-ID)"]
    end
```

### Storage and Persistence Model
- **SQLite (WAL Mode)**: Authoritative system of record for session history, turns, document metadata, and chunk text bodies (`outputs/sessions/assistant_memory.sqlite`).
- **Redis Stack**: High-performance vector database on port `6380` utilizing COSINE distance metric and Tag metadata filtering (`lab07:kb:*`).
- **Local LLM Inference**: Fully offline OpenAI-compatible REST server (LM Studio) hosted on `http://localhost:1234/v1`.

---

## 3. Scientific Evaluation & Benchmarking

The system is validated against a frozen **100-query scientific benchmark** (`evaluation/datasets/frozen_eval_100.json`) spanning 8 technical documents (4 university laboratory reports and 4 real-world engineering architecture specifications).

### Benchmark Results (100 Frozen Queries)

| Evaluation Metric | Target Threshold | Measured Baseline | Final Status |
|---|---|---|---|
| **Evidence Recall@1** | $\ge 0.6500$ | **0.7059** | **PASS** |
| **Evidence Recall@3** | $\ge 0.8000$ | **0.8000** | **PASS** |
| **Evidence Recall@5** | $\ge 0.8500$ | **0.9118** | **PASS** |
| **Hit@5** | $\ge 0.9200$ | **0.9882** | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.7500$ | **0.8182** | **PASS** |
| **Silent Truncation Violations** | $= 0$ | **0** | **PASS** |

### Benchmark Query Breakdown
- **Exact Technical Facts (25 queries)**: Exact parameters, model revisions, dataset shapes, and command flags.
- **Semantic Paraphrase (20 queries)**: Conceptual descriptions of system components, trade-offs, and architectures.
- **Cross-Document Comparison (20 queries)**: Contrasting findings across multiple documents with source attribution.
- **Numerical Reasoning (20 queries)**: Retrieving quantitative facts and executing verified arithmetic operations.
- **Unsupported Negative Controls (15 queries)**: Out-of-corpus queries verifying graceful abstention and refusal.
- **Linguistic Distribution**: 50 queries in Ukrainian, 50 queries in English.

---

## 4. Defect Resolution Matrix (Audit Traceability)

All 8 architectural and functional defects identified during the repository audit were resolved and verified with dedicated tests:

| Defect ID | Description & Severity | Root Cause | Implemented Resolution | Verification Artifact |
|---|---|---|---|---|
| **DEFECT-01** | `CRITICAL`: Fragile Coursework CLI / Missing Production API | Only a basic Streamlit script existed without REST contracts. | Implemented FastAPI REST application (`/api/v1`) with Pydantic request/response schemas, CORS, and request ID tracing. | `tests/contracts/test_api_contracts.py` |
| **DEFECT-02** | `CRITICAL`: Broken Ingestion Lifecycle & Scanned PDF Silently Accepted | Empty/scanned PDFs were silently indexed as valid empty documents; deletions did not clear Redis vectors. | Added PyMuPDF text validation raising `ExtractionError` for scanned files; implemented atomic Redis/SQLite deletion. | `tests/integration/test_document_lifecycle.py` |
| **DEFECT-03** | `CRITICAL`: Session Persistence Integrity & Concurrency | SQLite transactions lacked WAL mode, foreign keys, or turn isolation. | Enabled `PRAGMA journal_mode=WAL`, foreign key enforcement, transaction rollbacks, and idempotency keys. | `tests/contracts/test_session_contract.py` |
| **DEFECT-04** | `HIGH`: Fragile Agent Orchestration & Silent Arithmetic Failures | Small local LLMs occasionally generated math inline instead of calling the tool. | Implemented AST-sandboxed calculator with deterministic recovery that executes formulas and tags origin as `recovery_executed`. | `tests/unit/test_recovery.py` |
| **DEFECT-05** | `HIGH`: Overfitted Academic Corpus | Corpus contained only Labs 3–6, failing to represent production engineering knowledge. | Added 4 production engineering documents (ADR-001, Post-mortem, API Spec, Runbook) under `assets/knowledge/engineering/`. | `evaluation/baselines/benchmark_results.json` |
| **DEFECT-06** | `HIGH`: Underpowered Evaluation Suite | Historical evaluation had only 15 queries and used substring matching for metrics. | Constructed 100-query frozen benchmark with exact set overlap metrics, MRR, and zero-truncation verification. | `evaluation/datasets/frozen_eval_100.json` |
| **DEFECT-07** | `HIGH`: Broken CI Workflow & Offline Runner Incompatibility | CI had invalid `if: ${{ false }}` syntax and lacked dependency caches. | Repaired `labs-6-7-ci.yml` and `lab5-ci.yml`, added offline mocks for embeddings and Redis, enabling 100% green CI. | GitHub Actions Run ID `41861178786` |
| **DEFECT-08** | `MEDIUM`: Missing Observability & Tracing | No metrics, request tracing, or structured health monitoring. | Added Prometheus client (`/metrics`), JSON structured logger, and `/api/v1/health/live` & `/ready` endpoints. | `src/knowledge_copilot/observability/` |

---

## 5. Academic Traceability Matrix

The product preserves full academic compliance with the original university coursework requirements while elevating engineering execution to production standards:

| Coursework Requirement | Academic Objective | Production Implementation in Lab 7 | Verification Evidence |
|---|---|---|---|
| **REQ-01: Corpus Ingestion** | Ingest Markdown reports from Labs 3–6. | Extended ingestion service supporting Labs 3–6 + 4 production engineering documents with SHA-256 fingerprinting. | `assets/knowledge/`, `test_knowledge_base.py` |
| **REQ-02: Local RAG Pipeline** | Dense vector search using local embeddings and Redis. | Hybrid retrieval engine combining Multilingual-E5-small dense search and BM25 with Reciprocal Rank Fusion ($k=60$). | `src/knowledge_copilot/rag/`, `run_benchmark.py` |
| **REQ-03: Tool Integration** | Calculator tool for quantitative calculations. | AST-sandboxed arithmetic evaluator (`ast.parse`) with strict bounds checking and deterministic recovery. | `src/knowledge_copilot/agents/tools.py` |
| **REQ-04: Conversation Memory** | Multi-turn chat with persistent session memory. | SQLite WAL session repository with turn history, evidence logging, and session isolation. | `src/knowledge_copilot/infrastructure/sqlite_repo.py` |
| **REQ-05: Grounding & Citations** | Prevent hallucinations and attribute facts to sources. | Post-generation citation verifier enforcing `[filename:section]` format against current evidence chunks. | `src/knowledge_copilot/agents/verifier.py` |
| **REQ-06: Scientific Reporting** | Measure Recall and MRR across queries. | 100-query frozen evaluation suite exceeding all baseline targets (Recall@5 = 0.9118, MRR = 0.8182). | `evaluation/datasets/frozen_eval_100.json` |

---

## 6. REST API Reference

The FastAPI application provides OpenAPI documentation at `http://localhost:8000/docs`.

### 6.1 `POST /api/v1/chat`
Submit a question to the Copilot.

**Request**:
```json
{
  "query": "What was the peak memory consumption during the Redis OOM incident?",
  "session_id": "eng_session_01",
  "retrieval_mode": "hybrid",
  "top_k": 5
}
```

**Response**:
```json
{
  "session_id": "eng_session_01",
  "query": "What was the peak memory consumption during the Redis OOM incident?",
  "answer": "During the incident, Redis memory consumption peaked at 4.2 GB against a 4.0 GB limit [incident_postmortem_redis_oom.md:Incident Overview].",
  "evidence": [
    {
      "chunk_id": "postmortem_001",
      "document_id": "incident_postmortem_redis_oom",
      "filename": "incident_postmortem_redis_oom.md",
      "section": "Incident Overview",
      "text": "Redis memory consumption peaked at 4.2 GB...",
      "score": 0.0322,
      "retrieval_method": "hybrid"
    }
  ],
  "tool_calls": [],
  "turn_count": 2,
  "execution_time_ms": 1420.5,
  "confidence": "high",
  "trace_id": "req-9b8f2c"
}
```

### 6.2 `POST /api/v1/documents`
Upload and index a technical document (`.md`, `.txt`, `.pdf`).

**Request**: Multipart form-data with `file` and optional `collection`.

**Response**:
```json
{
  "document_id": "doc_a1b2c3d4",
  "filename": "architecture_spec.pdf",
  "status": "indexed",
  "chunk_count": 14,
  "size_bytes": 1048576,
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "collection": "default",
  "created_at": "2026-11-04T17:00:00Z"
}
```

### 6.3 `GET /api/v1/health/live` & `GET /api/v1/health/ready`
Liveness and readiness probes returning component status:
```json
{
  "status": "ready",
  "redis_connected": true,
  "sqlite_connected": true,
  "llm_available": true,
  "indexed_documents": 8,
  "version": "2.0.0"
}
```

---

## 7. Quickstart Installation and Deployment

### 7.1 Prerequisites
- Python 3.11+
- Docker Desktop (for Redis Stack)
- LM Studio (or local OpenAI-compatible inference server on port `1234`)

### 7.2 Step-by-Step Setup

```bash
# 1. Clone repository and navigate to Lab 7
cd labs/lab07_final_project

# 2. Start Redis Stack container (port 6380)
docker compose up -d lab07-redis

# 3. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Start local model in LM Studio
# Ensure model 'mistralai/ministral-3-3b' is loaded and server is started on port 1234.

# 5. Launch FastAPI application server (port 8000)
uvicorn src.knowledge_copilot.api.app:app --host 0.0.0.0 --port 8000

# 6. Launch Streamlit UI (port 8501) in a separate terminal
streamlit run src/knowledge_copilot/ui/streamlit_app.py
```

---

## 8. Verification and Quality Gates

Execute the full automated test suite (127 items spanning unit, contract, integration, and security tests):

```bash
# Run complete test suite:
pytest tests -v

# Run 100-query frozen benchmark:
python evaluation/run_benchmark.py --dataset evaluation/datasets/frozen_eval_100.json
```
