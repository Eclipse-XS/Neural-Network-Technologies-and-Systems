# Local Engineering Knowledge Copilot: Architecture Specification

## 1. Architectural Philosophy and Design Principles

The Local Engineering Knowledge Copilot is engineered as a **modular monolith** emphasizing strict domain isolation, deterministic execution paths, reproducible retrieval, and zero external cloud dependencies.

Key principles:
1. **Local-First Privacy**: All inference, vector storage, document parsing, and session persistence run strictly within the local host boundary (`localhost`), ensuring zero data exfiltration.
2. **Explicit Separation of Concerns**: Clean separation between API contracts, application services, domain models, retrieval-augmented generation (RAG) components, and persistence adapters.
3. **Dual-Store Persistence**:
   - **SQLite with WAL (Write-Ahead Logging)**: Authoritative system of record for session history, turns, document ingestion metadata, and chunk text bodies.
   - **Redis Stack Vector Database**: Ephemeral / reconstructible vector index powering sub-millisecond dense vector similarity search and metadata tag filtering.
4. **Bounded Agent Orchestration**: Maximum 10 execution turns per query, AST-sandboxed arithmetic evaluator, and post-generation citation verification to eliminate ungrounded hallucinations.

---

## 2. Component Taxonomy

```
labs/lab07_final_project/src/knowledge_copilot/
├── domain/                  # Pure business models, enums, custom exceptions
│   ├── models.py            # Pydantic v2 domain schemas (Document, Chunk, Session, Turn)
│   ├── enums.py             # RetrievalMode, DocumentStatus, EvidenceQuality
│   └── exceptions.py        # Domain-specific hierarchy (ExtractionError, etc.)
├── rag/                     # Retrieval-Augmented Generation subsystem
│   ├── extractors.py        # MIME detection, path sanitization, PyMuPDF parser
│   ├── splitters.py         # Markdown/PDF header-aware chunker with token bounding
│   ├── embeddings.py        # Multilingual-E5-small adapter with offline fallback
│   ├── bm25.py              # Rank-BM25 with token normalization & stopword filtering
│   └── fusion.py            # Reciprocal Rank Fusion (RRF, k=60)
├── infrastructure/          # Storage, networking, and external adapters
│   ├── sqlite_repo.py       # SQLite WAL repository with connection pooling
│   ├── redis_store.py       # Redis Stack vector store (HNSW/FLAT cosine metric)
│   └── llm_client.py        # Local OpenAI-compatible HTTP client (LM Studio)
├── agents/                  # Autonomous reasoning & safety boundaries
│   ├── tools.py             # AST-safe calculator and RAG tool wrappers
│   ├── orchestrator.py      # Bounded ReAct orchestration loop (max 10 turns)
│   ├── verifier.py          # Citation and claim grounding verification
│   └── recovery.py          # Deterministic arithmetic fallback executor
├── application/             # Core application use cases
│   ├── ingestion_service.py # Ingestion lifecycle, idempotency, failure rollback
│   ├── query_service.py     # Hybrid search routing, multi-target expansion
│   └── session_service.py   # Isolated multi-turn conversation memory
├── api/                     # REST presentation layer
│   ├── app.py               # FastAPI application factory & middleware
│   ├── dependencies.py      # Dependency injection container
│   ├── schemas.py           # HTTP request & response DTOs
│   └── routes/              # Modular routers (chat, documents, health)
├── observability/           # Telemetry and diagnostics
│   ├── logging.py           # Structured JSON logging with request-ID propagation
│   ├── metrics.py           # Prometheus counters, gauges, and histograms
│   └── tracer.py            # Latency and step duration tracker
└── ui/                      # Streamlit graphical interface
    └── streamlit_app.py     # Reactive UI consuming the application services
```

---

## 3. Data Contracts and Storage Model

### 3.1 SQLite Schema (`assistant_memory.sqlite`)

The SQLite database operates under `PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;`.

- **`documents`**:
  - `document_id` (TEXT PRIMARY KEY)
  - `filename` (TEXT NOT NULL)
  - `content_type` (TEXT NOT NULL)
  - `sha256` (TEXT NOT NULL)
  - `size_bytes` (INTEGER NOT NULL)
  - `status` (TEXT NOT NULL: `processing`, `indexed`, `failed`)
  - `chunk_count` (INTEGER DEFAULT 0)
  - `collection` (TEXT DEFAULT 'default')
  - `error_message` (TEXT)
  - `created_at`, `updated_at` (TEXT ISO 8601)

- **`chunks`**:
  - `chunk_id` (TEXT PRIMARY KEY)
  - `document_id` (TEXT NOT NULL, FK -> documents)
  - `ordinal` (INTEGER NOT NULL)
  - `text` (TEXT NOT NULL)
  - `section` (TEXT)
  - `token_count` (INTEGER NOT NULL)
  - `sha256` (TEXT NOT NULL)
  - `evidence_quality` (TEXT NOT NULL)
  - `metadata_json` (TEXT NOT NULL)

- **`sessions`**:
  - `session_id` (TEXT PRIMARY KEY)
  - `title` (TEXT NOT NULL)
  - `created_at`, `updated_at` (TEXT ISO 8601)

- **`turns`**:
  - `turn_id` (TEXT PRIMARY KEY)
  - `session_id` (TEXT NOT NULL, FK -> sessions)
  - `user_query` (TEXT NOT NULL)
  - `assistant_response` (TEXT NOT NULL)
  - `evidence_json` (TEXT)
  - `tools_json` (TEXT)
  - `turn_index` (INTEGER NOT NULL)
  - `created_at` (TEXT ISO 8601)

### 3.2 Redis Vector Schema (`lab07:kb:*`)

Redis Stack stores chunk embeddings as Hashes indexed by RediSearch index `lab07_course_kb_idx`:
- `vector`: FLOAT32 array (384 dimensions for `intfloat/multilingual-e5-small`) with COSINE distance metric.
- Tag fields: `source_id`, `document_id`, `filename`, `section`, `chunk_id`, `collection`, `evidence_quality`.

---

## 4. Bounded Agent Reasoning and Safe Arithmetic

1. **AST Arithmetic Sandbox**:
   - `eval()` and `exec()` are strictly banned.
   - The expression tree is parsed using Python's `ast.parse(mode='eval')`.
   - Allowed nodes: `ast.Expression`, `ast.BinOp`, `ast.UnaryOp`, `ast.Constant`, `ast.Add`, `ast.Sub`, `ast.Mult`, `ast.Div`, `ast.FloorDiv`, `ast.Mod`, `ast.USub`, `ast.UAdd`.
   - Bounds: operands are validated against `abs(x) < 1e15` and exponentiation (`**`) is rejected to prevent OOM/DoS attacks.
2. **Deterministic Recovery**:
   - If an agent generates an arithmetic formula in its reasoning chain without invoking the calculator tool, the deterministic recovery module detects arithmetic patterns (`r"(\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?)"`), executes the AST calculator, and tags the tool execution as `origin: recovery_executed`.
3. **Evidence Verifier**:
   - Analyzes generated responses for citation format `[filename.ext:section]`.
   - Checks that all cited claims match actual tokens in retrieved evidence chunks.
   - Flags suspicious placeholder citations (e.g., `[source:1]`, `[unknown]`) as ungrounded.

---

## 5. Observability and Monitoring

- **Prometheus Metrics Endpoint**: `/metrics`
  - `rag_requests_total{endpoint, status}`: Total HTTP requests.
  - `rag_request_duration_seconds{endpoint}`: Request latency histogram.
  - `rag_indexed_documents`: Gauge of indexed documents.
  - `rag_indexed_chunks`: Gauge of indexed text chunks.
  - `rag_ingestion_failures_total{error_type}`: Counter of ingestion errors.
  - `agent_turn_count`: Histogram of reasoning turns per query.
  - `agent_tool_executions_total{tool_name, status}`: Tool invocation counter.
- **Structured Tracing**: Request IDs (`X-Request-ID`) and response times (`X-Response-Time-MS`) injected on all API endpoints.
