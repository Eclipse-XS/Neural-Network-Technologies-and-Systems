# API Specification and Data Contracts: Knowledge Copilot v1

## Base URI
`/api/v1`

## Authentication & Headers
- All requests should include `X-Request-ID` for end-to-end request tracing.
- If omitted, the FastAPI middleware automatically injects a UUIDv4 string into response headers.
- Rate limit: 60 requests per minute per IP for `/chat/completions`; 10 concurrent requests for `/documents/upload`.

## Core Endpoints

### 1. Health Checks
- `GET /health/live`: Returns `{"status": "ok"}` when the ASGI server is accepting connections (HTTP 200).
- `GET /health/ready`: Returns `{"status": "ready", "redis": true, "database": true}` (HTTP 200), or HTTP 503 if dependencies fail.
- `GET /metrics`: Prometheus metric scrape endpoint in OpenMetrics exposition format.

### 2. Document Management
- `GET /documents`: Lists all registered documents with status (`PENDING`, `PROCESSING`, `INDEXED`, `FAILED`).
- `POST /documents/upload`: Multipart file upload accepting `.md`, `.txt`, `.pdf`. Returns `DocumentUploadResponse`.
  - Max file size: 20 MB.
  - Rejection policy: Zero-byte files or corrupted PDFs without an extractable text layer return HTTP 400 with `ExtractionError`.
- `DELETE /documents/{document_id}`: Deletes document record, associated SQLite chunks, and Redis vector indexes atomically.

### 3. Agentic Chat & Retrieval
- `POST /chat/completions`: Executes a query turn with grounded RAG and agentic verification.
  - Request Body:
    ```json
    {
      "query": "What is the Hit@5 baseline in lab03?",
      "session_id": "optional-uuid",
      "retrieval_mode": "hybrid",
      "top_k": 5
    }
    ```
  - Response Body (`ChatResponse`):
    - `answer`: string
    - `grounding_status`: `GROUNDED` | `PARTIALLY_GROUNDED` | `UNSUPPORTED` | `INSUFFICIENT_EVIDENCE`
    - `citations`: list of `Citation` objects with source document, line/section, and quote
    - `tool_executions`: list of `ToolExecutionRecord` with origin (`MODEL_REQUESTED` vs `RECOVERY_EXECUTED`)
    - `execution_metadata`: execution time, token count, and retrieval latency in milliseconds.
