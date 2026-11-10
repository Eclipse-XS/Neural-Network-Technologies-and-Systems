# ADR 001: Vector Storage and Relational Persistence Architecture

## Status
Accepted

## Context
The Local Engineering Knowledge Copilot requires persistent, high-throughput hybrid retrieval (dense vector embeddings combined with lexical BM25 indexing) as well as deterministic, ACID-compliant session and document state management.

Key operational constraints:
1. 100% offline, local-first execution on developer workstations (Windows/Linux).
2. Low query latency (<50ms for retrieval pipeline).
3. Robust crash recovery and deterministic session history without external database server daemons when running lightweight tests.
4. Scale target: up to 100,000 document chunks per local knowledge base.

## Decision
We adopt a hybrid persistence model:
1. **Redis Stack (Search & JSON)** as the vector database:
   - Utilizes `HNSW` (Hierarchical Navigable Small World) and `FLAT` vector indexing with `FLOAT32` precision.
   - Cosine distance metric for normalized embedding vectors (`multilingual-e5-base`, dimension = 768).
   - Fast atomic prefix-based key deletion and namespace isolation (`lab07:kb:*`).
2. **SQLite with WAL (Write-Ahead Logging)** for application state and chat memory:
   - Embedded storage with zero external network overhead.
   - Enforces `PRAGMA busy_timeout = 5000` to prevent database locks under concurrent API accesses.
   - Guarantees strict transactional integrity for session turns, document status transitions (`PENDING -> PROCESSING -> INDEXED`), and user feedback.

## Alternatives Considered
- **pgvector (PostgreSQL)**: Rejected due to heavy container footprint and complex local installation on developer workstations.
- **Milvus / Qdrant**: High memory overhead (>=2 GB idle RAM), exceeding the target budget of local laptop execution.
- **ChromaDB**: Inconsistent SQLite locking under multi-threaded FastAPI runners and frequent schema breakages across releases.

## Consequences
- **Positive**: Extremely fast sub-15ms vector query latency; robust transaction isolation; trivial backup by copying SQLite `.db` file and Redis `.rdb` snapshot.
- **Negative**: Requires Redis Stack Docker container or service on port 6380 for live vector operations; fallback to in-memory/mock adapters is required for zero-dependency CI runs.
