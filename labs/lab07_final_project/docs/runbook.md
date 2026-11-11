# Operations and Troubleshooting Runbook

## 1. System Requirements and Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 22.04+).
- **Python Runtime**: Python 3.11, 3.12, or 3.13 (Python 3.13 tested).
- **Docker Desktop**: Required for running the Redis Stack container (providing RediSearch and RedisJSON vector indexing).
- **Local Inference Server**: LM Studio (or Ollama / llama.cpp) running on port `1234` with an OpenAI-compatible `/v1` endpoint.

---

## 2. Architecture & Entrypoint Separation

| Subsystem | Modern Implementation (CANONICAL) | Historical Implementation (LEGACY COMPATIBILITY) |
|---|---|---|
| **Core Architecture** | Modular monolith under `src/knowledge_copilot/` | Monolithic script under `src/` (`agent.py`, `knowledge_base.py`) |
| **REST API Server** | `labs.lab07_final_project.src.knowledge_copilot.api.app:app` | None (CLI-only) |
| **Web UI** | `labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py` | `labs/lab07_final_project/web_app.py` |
| **CLI Assistant** | Interactive API / Streamlit interface | `labs/lab07_final_project/app.py` |
| **Storage Architecture** | Dual storage: SQLite WAL (`assistant_memory.sqlite`) + Redis Stack (Port `6380`) | Dual storage: SQLite + Redis Stack (Port `6380`) |
| **Retrieval Engine** | Hybrid Dense (multilingual-e5-small) + BM25 with Reciprocal Rank Fusion ($k=60$) | Dense-only (all-MiniLM-L6-v2) |

---

## 3. Quickstart Execution Guide

### 3.1 Step 1: Start Redis Stack Service

From the `labs/lab07_final_project/` directory:

```bash
cd labs/lab07_final_project
docker compose up -d redis
```

> **Note on Service Name**: The docker compose service is named `redis` (container name `lab07-redis`). Use `docker compose up -d redis` or simply `docker compose up -d`. Do **not** run `docker compose up -d lab07-redis` as Compose expects the service identifier.

Verify Redis is responding to health probes on port `6380`:
```bash
docker exec -it lab07-redis redis-cli -p 6379 ping
# Expected output: PONG
```

### 3.2 Step 2: Configure and Start Local LM Studio

1. Open **LM Studio**.
2. Download or load a local model that natively supports function calling / tool calling (e.g. `mistralai/ministral-3-3b`, `qwen2.5-7b-instruct`, or `hermes-3-llama-3.2-3b`).
3. Navigate to the **Developer / Local Server** tab (`<->`).
4. Select the loaded model and toggle **Start Server** on port `1234` (`http://localhost:1234/v1`).
5. Ensure CORS is enabled.

### 3.3 Step 3: Initialize Virtual Environment

From the repository root or the lab directory:

```powershell
# Using the dedicated virtual environment:
cd labs/lab07_final_project
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.4 Step 4: Launch FastAPI Application Server

Run from the repository root:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m uvicorn labs.lab07_final_project.src.knowledge_copilot.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Or from `labs/lab07_final_project/` with `PYTHONPATH`:

```powershell
cd labs/lab07_final_project
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m uvicorn knowledge_copilot.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify health and readiness probes:

```powershell
curl http://localhost:8000/api/v1/health/live
# Expected: {"status":"live","service":"knowledge_copilot","timestamp":"..."}

curl http://localhost:8000/api/v1/health/ready
# Expected: {"status":"ready","redis_connected":true,"sqlite_connected":true,"llm_available":true,...}
```

### 3.5 Step 5: Launch Streamlit Web Interface

Run from the repository root:

```powershell
.\labs\lab07_final_project\.venv\Scripts\streamlit.exe run labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py --server.port 8501
```

Access the application in your browser at `http://localhost:8501`.

---

## 4. Operational Tasks

### 4.1 Ingest Knowledge Corpus

To ingest the laboratory reports and engineering specifications:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -c "from labs.lab07_final_project.src.knowledge_copilot.api.dependencies import get_ingestion_service; from pathlib import Path; s = get_ingestion_service(); [s.ingest_file(f.name, f.read_bytes(), document_id=f.stem) for f in Path('labs/lab07_final_project/assets/knowledge').glob('**/*.*') if f.suffix in {'.md', '.pdf', '.txt'}]; print('Ingestion completed successfully.')"
```

### 4.2 Execute Test Suite

```powershell
# Full Lab 7 test suite (232 tests) with live Redis running on port 6380:
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests -q
```

### 4.3 Run Frozen Scientific Benchmark

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe labs/lab07_final_project/evaluation/run_benchmark.py --dataset labs/lab07_final_project/evaluation/datasets/frozen_eval_100.json
```

---

## 5. Diagnostics and Troubleshooting

| Symptom | Probable Cause | Remediation Step |
|---|---|---|
| **Health endpoint reports `redis_connected: false`** | Redis container stopped or port conflict on 6380 | Run `docker compose ps` to inspect container status. Check if port 6380 is occupied. |
| **`llm_available: false`** | LM Studio server not started or bound to different port | Ensure LM Studio "Local Server" toggle is ON and bound to `http://localhost:1234`. |
| **PDF upload fails with `Scanned or image-only PDF`** | Uploaded PDF has no digital text layer | Run OCR on the PDF before uploading. System intentionally rejects empty scans with `ExtractionError`. |
| **`sqlite3.OperationalError: database is locked`** | Long-running transaction or non-WAL mode | Check `PRAGMA journal_mode;` in SQLite. Ensure WAL mode is active. |
| **`ZeroDivisionError` in calculator** | Unsafe arithmetic query submitted | Handled gracefully by AST calculator; returns structured error without crashing. |
| **`IndexingError: Vector store unreachable`** | Redis Stack service was not started before ingestion | Start Redis container via `docker compose up -d redis`. |
