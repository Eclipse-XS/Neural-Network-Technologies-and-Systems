# Local Engineering Knowledge Copilot

[![CI: Labs 6 & 7 Offline Suite](https://github.com/Eclipse-XS/Neural-Network-Technologies-and-Systems/actions/workflows/labs-6-7-ci.yml/badge.svg)](https://github.com/Eclipse-XS/Neural-Network-Technologies-and-Systems/actions/workflows/labs-6-7-ci.yml)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI: 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit: 1.42+](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![VectorDB: Redis Stack](https://img.shields.io/badge/VectorDB-Redis%20Stack%20(Port%206380)-DC382D.svg)](https://redis.io)
[![Inference: Local LM Studio](https://img.shields.io/badge/Inference-LM%20Studio%20(Port%201234)-8A2BE2.svg)](https://lmstudio.ai)

A local-first engineering prototype and reproducible research system designed to ingest technical documentation, perform hybrid dense-lexical retrieval with Reciprocal Rank Fusion, reason over multi-document evidence, execute safe AST-sandboxed calculations, and persist isolated multi-turn conversations without external cloud dependencies.

![Recall Comparison](artifacts/portfolio/retrieval_recall_comparison.png)
*Figure 1: Retrieval Recall@K across Dense, BM25 Lexical, and Hybrid RRF modes on the 100-query frozen benchmark.*

---

## 1. Project Overview

### Problem Statement
General-purpose Large Language Models (LLMs) frequently hallucinate quantitative values, conflate separate technical specifications, and fail to provide exact provenance when answering complex engineering queries. Relying on commercial cloud APIs introduces vendor lock-in, recurring operational cost, and severe privacy risks for proprietary codebases and confidential technical reports.

### Solution: Local Engineering Knowledge Copilot
The Copilot is engineered as a privacy-preserving, local-first RAG (Retrieval-Augmented Generation) and agentic assistant. It runs entirely on localhost using consumer workstation hardware:
- **Corpus Grounding**: Answers are synthesized strictly from indexed Markdown, plain text, and PDF engineering documents.
- **Auditable Provenance**: Every factual claim is paired with exact document filenames and chunk identifiers (`[filename#chunk-id]`).
- **Deterministic Math**: Mathematical operations are delegated to an AST-sandboxed calculator tool rather than relying on probabilistic next-token prediction.
- **Graceful Abstention**: When evidence is absent from the indexed corpus, the system explicitly refuses to answer rather than fabricating plausible misinformation.

---

## 2. Key Features

- **Document Ingestion Lifecycle**: Supports Markdown, plain text, and digital PDFs with structure-aware chunking (hierarchical headers, token-bounded sub-splitting to $\le 240$ tokens, SHA-256 deduplication, and scanned/empty PDF rejection via `ExtractionError`).
- **Hybrid Retrieval Engine**: Combines 384-dimensional dense semantic embeddings (`multilingual-e5-small`) with BM25 lexical token matching via Reciprocal Rank Fusion (RRF).
- **Source-Grounded Answering & Citation Gate**: Enforces strict evidence verification; answers lacking valid citations or evidence backing are flagged and refused.
- **AST-Sandboxed Calculator Tool**: Evaluates arithmetic expressions safely through Python's `ast` module, preventing code injection while guaranteeing mathematical correctness.
- **Isolated Multi-Turn Session Memory**: Thread-safe SQLite persistence with Write-Ahead Logging (WAL) and strict session isolation across distinct user sessions.
- **Modular Monolith Architecture**: Decoupled domain, application, RAG, agent, infrastructure, and presentation layers exposed via FastAPI REST endpoints and an interactive Streamlit UI.
- **Telemetry & Health Probes**: `/api/v1/health/live` and `/ready` probes monitoring Redis, SQLite, and LM Studio connectivity, complemented by Prometheus `/metrics` and structured JSON logs.

---

## 3. System Architecture

```mermaid
flowchart TD
    User["Engineering User / Web Browser"] -->|HTTP / UI| Streamlit["Streamlit UI (Port 8501)"]
    User -->|REST API Requests| FastAPI["FastAPI Server (Port 8000)"]
    Streamlit -->|HTTP Client / Services| FastAPI

    subgraph MonolithCore ["Knowledge Copilot Core"]
        FastAPI --> IngestSvc["Application: IngestionService"]
        FastAPI --> QuerySvc["Application: QueryService"]
        FastAPI --> SessionSvc["Application: SessionService"]

        QuerySvc --> Agent["Agent: AgentOrchestrator (ReAct <= 10 turns)"]
        Agent --> CalcTool["Tool: AST Calculator (Safe Math)"]
        Agent --> Verifier["Evidence Verifier & Citations Gate"]

        QuerySvc --> HybridEngine["RAG: Hybrid Engine (BM25 + Dense RRF)"]
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
- **SQLite (WAL Mode)**: System of record for conversation sessions, turns, document metadata, and chunk text bodies (`outputs/sessions/assistant_memory.sqlite`).
- **Redis Stack**: In-memory vector database on port `6380` utilizing COSINE distance metric and tag metadata filtering (`lab07:kb:*`).
- **Local LLM Inference**: Fully offline OpenAI-compatible REST server (LM Studio) hosted on `http://localhost:1234/v1`.

### Canonical vs. Legacy Implementation

| Subsystem | Modern Implementation (CANONICAL) | Historical Implementation (LEGACY COMPATIBILITY) |
|---|---|---|
| **Directory** | `src/knowledge_copilot/` | `src/` (`agent.py`, `knowledge_base.py`, `experiments.py`) |
| **REST Server** | `src/knowledge_copilot/api/app.py` | None |
| **Web UI** | `src/knowledge_copilot/ui/streamlit_app.py` | `web_app.py` |
| **CLI App** | Streamlit / REST API | `app.py` |
| **Retrieval** | Hybrid RRF: `multilingual-e5-small` + BM25 | Dense-only: `all-MiniLM-L6-v2` |

---

## 4. How It Works

```mermaid
sequenceDiagram
    autonumber
    actor User as Engineer / User
    participant UI as Streamlit UI / API
    participant Orch as AgentOrchestrator
    participant RAG as Hybrid RAG Engine
    participant Calc as AST Calculator
    participant Verifier as Evidence Verifier
    participant LLM as Local LM Studio

    User->>UI: Submit Question ("What is the difference in runs between Lab 5 and Lab 4?")
    UI->>Orch: Dispatch query with session_id

    Orch->>LLM: ReAct prompt with tools (search_knowledge_base, calculate)
    LLM-->>Orch: Tool Call: search_knowledge_base(query="Lab 4 Lab 5 runs")

    Orch->>RAG: Hybrid search across Redis & BM25
    RAG-->>Orch: Evidence chunks [lab04#chunk-1], [lab05#chunk-1]

    Orch->>LLM: Provide evidence to context
    LLM-->>Orch: Tool Call: calculate(expression="17 - 13")

    Orch->>Calc: Execute AST math safely
    Calc-->>Orch: Result: 4

    Orch->>LLM: Provide calculation result
    LLM-->>Orch: Final answer with citations

    Orch->>Verifier: Audit claims against retrieved citations
    Verifier-->>Orch: Citations Verified (Grounded)

    Orch-->>UI: Authoritative answer + source citations + tool trace
    UI-->>User: Render grounded response with inspectable evidence
```

---

## 5. Technology Stack

| Layer | Component | Version / Specification |
|---|---|---|
| **Runtime & Language** | Python | 3.11 / 3.12 / 3.13 (Python 3.13 tested) |
| **Vector Store** | Redis Stack | Redis 7.4.0 (RediSearch + RedisJSON, port `6380`) |
| **Relational Storage** | SQLite 3 | WAL mode, thread-safe session & chunk persistence |
| **Dense Embeddings** | Hugging Face Transformers | `intfloat/multilingual-e5-small` (384 dimensions) |
| **Lexical Engine** | Rank-BM25 | Parameterized BM25 ($k_1=1.5, b=0.75$) with NLTK tokenization |
| **Rank Fusion** | Reciprocal Rank Fusion | RRF algorithm combining dense semantic and BM25 lexical rankings |
| **API Framework** | FastAPI | 0.115+, Pydantic v2 schemas, Starlette middleware |
| **Frontend UI** | Streamlit | 1.42+, wide layout, multi-session chat interface |
| **Local Inference Host** | LM Studio | OpenAI-compatible REST API on port `1234` |
| **Test Framework** | Pytest | 9.1.1 (185 automated tests across unit, contract, integration, security) |

---

## 6. Quickstart

### Prerequisites
1. Docker Desktop installed and running.
2. LM Studio running with a function-calling model on port 1234 (`http://localhost:1234/v1`).

### Step 1: Start Redis Stack Service
```bash
# From repository root:
docker compose -f labs/lab07_final_project/docker-compose.yml up -d redis
```

### Step 2: Install Dependencies
```powershell
cd labs/lab07_final_project
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Launch FastAPI Backend Server
```powershell
# From repository root:
.\labs\lab07_final_project\.venv\Scripts\python.exe -m uvicorn labs.lab07_final_project.src.knowledge_copilot.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Launch Streamlit Web Interface
```powershell
# From repository root:
.\labs\lab07_final_project\.venv\Scripts\streamlit.exe run labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py --server.port 8501
```

Access the UI at `http://localhost:8501`.

---

## 7. Example Interaction

### User Query
> *"What is the difference between the number of verified runs in Lab 5 and the number of generations in Lab 4?"*

### Tool Invocation Trace
```json
[
  {
    "event_type": "tool_call",
    "tool_name": "search_knowledge_base",
    "tool_arguments": {"query": "Lab 5 verified runs Lab 4 generations count"}
  },
  {
    "event_type": "tool_result",
    "tool_name": "search_knowledge_base",
    "result": {
      "chunks": [
        {
          "citation": "[lab05_multimodal_report.md#chunk-0]",
          "text": "Матриця містить 10 baseline-питань... 19 експериментальних рядків відповідають 17 фізичним генераціям."
        },
        {
          "citation": "[lab04_stable_diffusion_report.md#chunk-0]",
          "text": "Зафіксовано 13 унікальних генерацій разом із параметрами, часом виконання..."
        }
      ]
    }
  },
  {
    "event_type": "tool_call",
    "tool_name": "calculate",
    "tool_arguments": {"expression": "17 - 13"}
  },
  {
    "event_type": "tool_result",
    "tool_name": "calculate",
    "result": {"result": 4.0, "status": "SUCCESS"}
  }
]
```

### Synthesized Response
> "Based on the verified laboratory documentation, Lab 5 recorded **17** physical multimodal inference runs [[lab05_multimodal_report.md#chunk-0]](file:///labs/lab05_multimodal/report/report.md), while Lab 4 recorded **13** unique diffusion image generations [[lab04_stable_diffusion_report.md#chunk-0]](file:///labs/lab04_stable_diffusion/report/report.md). The difference is **4** runs, verified via AST calculation ($17 - 13 = 4$)."

---

## 8. Experimental Evaluation & Scientific Benchmarking

The system was evaluated against a frozen **100-query scientific benchmark** (`evaluation/datasets/frozen_eval_100.json`) spanning 8 technical documents (4 laboratory reports and 4 architecture/operational specifications).

### Distinguishing CI Pass from Retrieval Quality Targets

- **Infrastructure & Automated Test Suite: PASS (185/185 tests)**: All unit, integration, contract, security, schema, and API tests pass cleanly in CI and local execution.
- **Retrieval Quality Targets: FAIL (Sub-target baseline)**: Under strict passage-level evidence span verification ($\ge 75\%$ token overlap), ambitious retrieval thresholds were not met. We report these empirical findings transparently without fabricating scores.

### 100-Query Frozen Benchmark Results

Data drawn directly from authoritative artifact [`evaluation/baselines/benchmark_results.json`](evaluation/baselines/benchmark_results.json):

| Metric | Target Threshold | Dense Only | Lexical (BM25) | Hybrid (Dense + BM25 RRF) | Quality Status |
|---|---|---|---|---|---|
| **Evidence Recall@1** | $\ge 0.6500$ | 0.4176 | **0.4882** | 0.4363 | **Sub-target (FAIL)** |
| **Evidence Recall@3** | $\ge 0.8000$ | 0.6588 | **0.6647** | 0.6103 | **Sub-target (FAIL)** |
| **Evidence Recall@5** | $\ge 0.8500$ | 0.7647 | **0.7765** | 0.7020 | **Sub-target (FAIL)** |
| **Hit@5** | $\ge 0.9200$ | 0.8235 | **0.8588** | 0.8000 | **Near-target (FAIL)** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.7500$ | 0.5933 | **0.6584** | 0.6482 | **Sub-target (FAIL)** |
| **Silent Truncation Violations** | $= 0$ | **0** | **0** | **0** | **PASS** |

![MRR and Hit@5 Comparison](artifacts/portfolio/retrieval_mrr_hit_comparison.png)
*Figure 2: MRR and Hit@5 across retrieval modes.*

### Category Breakdown Analysis (Hybrid Mode)

![Category Breakdown](artifacts/portfolio/category_performance_breakdown.png)
*Figure 3: Hybrid retrieval performance across query complexity categories.*

- **Numerical Reasoning ($n=20$)**: **Hit@5 = 1.0000**, **MRR = 0.9250**, **Recall@5 = 0.9625**. Highest performing category due to distinct numeric tokens.
- **Exact Technical Facts ($n=25$)**: **Hit@5 = 0.8400**, **MRR = 0.7413**, **Recall@5 = 0.7567**. Strong lexical symbol matching.
- **Semantic Paraphrase ($n=20$)**: **Hit@5 = 0.7000**, **MRR = 0.4617**, **Recall@5 = 0.6250**. Dense vectors bridge cross-lingual phrasing.
- **Cross-Document Comparison ($n=20$)**: **Hit@5 = 0.6500**, **MRR = 0.4417**, **Recall@5 = 0.4500**. Challenging due to multi-target document quota competition.
- **Negative Controls ($n=15$)**: 100% retrieval-level abstention (no irrelevant chunks retrieved above threshold). In live inference, model answers require independent verification as tested in live scenarios.

### Latency Profile

![Latency Distribution](artifacts/portfolio/retrieval_latency_distribution.png)
*Figure 4: Retrieval latency percentiles (P50, Mean, P90, P95) measured on CPU.*

- **BM25 Lexical**: P50 = **2.12 ms**, P95 = **3.48 ms**, Mean = **2.24 ms**.
- **Dense Search**: P50 = **44.11 ms**, P95 = **89.12 ms**, Mean = **53.86 ms**.
- **Hybrid (Dense + BM25)**: P50 = **47.78 ms**, P95 = **98.86 ms**, Mean = **57.30 ms**.

---

## 9. Reproducibility

### Fast Offline Test Suite
```powershell
# Run the 185 tests in the Lab 7 virtual environment:
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests -q
```
Expected result: `185 passed, 2 skipped in ~120s`.

### Live Acceptance Suite
The live acceptance suite tests the complete 8-step lifecycle against live Redis and LM Studio:
```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests/e2e/test_live_lm_studio_acceptance.py -v
```

---

## 10. Limitations

- **Retrieval Threshold Gap**: Recall@1 (0.4363) and MRR (0.6482) on the 100-query benchmark remain below the ambitious targets ($\ge 0.6500$ and $\ge 0.7500$). Fine-tuning dense embeddings or implementing a cross-encoder reranker would improve these figures.
- **Cross-Document Competition**: When a query requires evidence from two distinct documents, RRF top-5 ranking frequently biases toward the single document with the highest dense similarity, leaving the second document unretrieved in top-5.
- **Local Model Capacity**: Quantized 3B models occasionally attempt mental arithmetic before falling back to the AST calculator. Prompt hardening and deterministic recovery logic mitigate but do not entirely eliminate this behavior.
- **Hardware Dependency**: While offline tests run on CPU, full live execution requires a host machine capable of running Docker and hosting a local LLM in LM Studio.

---

## 11. Documentation

- **[Primary Notebook (`notebooks/01_final_project.ipynb`)](notebooks/01_final_project.ipynb)**: Complete 22-step educational walkthrough of the modern Knowledge Copilot system.
- **[Academic Report (`report/report.md`)](report/report.md)**: Formal academic report in Ukrainian covering all 11 comprehensive chapters.
- **[Oral Defense Notes (`report/defense_notes.md`)](report/defense_notes.md)**: Targeted preparation for oral defense and committee examination.
- **[Detailed System Architecture (`docs/architecture.md`)](docs/architecture.md)**: Architectural specifications, data models, and C4 container design.
- **[Operations Runbook (`docs/runbook.md`)](docs/runbook.md)**: Step-by-step operations, setup, and troubleshooting guide.
- **[Scientific Evaluation Specification (`docs/evaluation.md`)](docs/evaluation.md)**: Authoritative documentation of the 100-query benchmark methodology.
- **[System Limitations Analysis (`docs/limitations.md`)](docs/limitations.md)**: Comprehensive engineering analysis of constraints and failure modes.
