# System Limitations and Engineering Boundaries

## 1. Document Extraction and Ingestion Boundaries

### 1.1 Scanned or Image-Only PDFs
- **Limitation**: The system relies on PyMuPDF (`fitz`) for direct text extraction. Scanned PDFs containing only bitmap images without an underlying OCR text layer cannot be parsed directly.
- **Handling**: The system **strictly refuses** to silently index empty documents as successful. It raises an explicit `ExtractionError` ("Scanned or image-only PDF without extractable text. OCR required.") and flags the document in the database with status `FAILED` and an error description.
- **Mitigation/Workaround**: Pre-process scanned documents using an external OCR pipeline (such as Tesseract or AWS Textract) prior to uploading to the Copilot.

### 1.2 File Size and Format Constraints
- Maximum file size is strictly capped at **10 MB** to prevent denial-of-service via resource exhaustion.
- Supported initial file extensions: `.md`, `.markdown`, `.txt`, `.pdf`. Binary executables, archives, and unknown formats are rejected with HTTP 400 Bad Request.

---

## 2. Local Model and Context Boundaries

### 2.1 Small Quantized LLM Behavioral Variance
- **Limitation**: The application is configured to run against quantized local models (e.g. `ministral-3-3b` or `qwen1.5-1.8b-chat`) via LM Studio. Small quantized models occasionally generate arithmetic inline without formally invoking the registered function tool.
- **Handling**: The application includes a deterministic recovery module (`src/knowledge_copilot/agents/recovery.py`) that extracts mathematical expressions and executes the AST calculator tool automatically, tagging the execution as `origin: recovery_executed`.

### 2.2 Token Budget and Context Window
- Local LLM context windows typically range between 4,096 and 8,192 tokens.
- To prevent prompt truncation and token overflow, chunks are strictly bounded to $\le 240$ tokens, and hybrid search defaults to retrieving the top $k=5$ evidence chunks.

---

## 3. Storage and Concurrency Limits

### 3.1 SQLite WAL Concurrency
- SQLite in WAL (Write-Ahead Logging) mode allows unlimited concurrent readers but serializes write transactions.
- Concurrency limit: suitable for single-user or small team desktop deployments (up to 50 concurrent reading sessions). High-throughput concurrent write environments ($>500$ writes/sec) would necessitate migrating the metadata repository to PostgreSQL.

### 3.2 Redis In-Memory Vector Storage
- Redis Stack stores vector indexes in RAM. At 384 dimensions with FLAT indexing, each chunk vector consumes approximately 1.5 KB of memory.
- For corpora larger than 1,000,000 chunks, HNSW indexing and Redis memory bounds must be actively monitored to prevent OOM errors (as detailed in `incident_postmortem_redis_oom.md`).
