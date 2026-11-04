# Operations and Troubleshooting Runbook

## 1. System Requirements and Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 22.04+).
- **Python Runtime**: Python 3.11, 3.12, or 3.13.
- **Docker Desktop**: Required for running the Redis Stack container.
- **Local Inference Server**: LM Studio (or Ollama / llama.cpp) running on port `1234`.

---

## 2. Quickstart Execution Guide

### 2.1 Step 1: Start Redis Stack
```bash
cd labs/lab07_final_project
docker compose up -d lab07-redis
```
Verify Redis is responding:
```bash
docker exec -it lab07-redis redis-cli -p 6379 ping
# Expected output: PONG
```

### 2.2 Step 2: Configure and Start LM Studio
1. Open **LM Studio**.
2. Load a supported model (e.g., `mistralai/ministral-3-3b` or `qwen/qwen1.5-1.8b-chat`).
3. Start the Local Server on port `1234` with CORS enabled.

### 2.3 Step 3: Initialize Virtual Environment
```bash
cd labs/lab07_final_project
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2.4 Step 4: Launch FastAPI Application Server
```bash
uvicorn src.knowledge_copilot.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Verify health endpoints:
```bash
curl http://localhost:8000/api/v1/health/live
curl http://localhost:8000/api/v1/health/ready
```

### 2.5 Step 5: Launch Streamlit Web Interface
```bash
streamlit run src/knowledge_copilot/ui/streamlit_app.py --server.port 8501
```
Open your browser to `http://localhost:8501`.

---

## 3. Operational Tasks

### 3.1 Ingest Knowledge Corpus
To ingest the default course laboratory reports and engineering documents into the Copilot:
```bash
python -c "from src.knowledge_copilot.api.dependencies import get_ingestion_service; from pathlib import Path; s = get_ingestion_service(); [s.ingest_file(f.name, f.read_bytes(), document_id=f.stem) for f in Path('assets/knowledge').glob('**/*.*') if f.suffix in {'.md', '.pdf', '.txt'}]; print('Ingestion completed successfully.')"
```

### 3.2 Execute Test Suite
```bash
# Run unit, contract, integration, and security tests:
pytest tests -v
```

### 3.3 Execute Scientific Benchmark
```bash
python evaluation/run_benchmark.py --dataset evaluation/datasets/frozen_eval_100.json
```

---

## 4. Diagnostics and Troubleshooting

| Symptom | Probable Cause | Remediation Step |
|---|---|---|
| **Health endpoint reports `redis_connected: false`** | Redis container stopped or port conflict on 6380 | Run `docker compose ps` to inspect container status. Check if port 6380 is occupied. |
| **`llm_available: false`** | LM Studio server not started or bound to different port | Ensure LM Studio "Local Server" toggle is ON and bound to `http://localhost:1234`. |
| **PDF upload fails with `Scanned or image-only PDF`** | Uploaded PDF has no digital text layer | Run OCR on the PDF before uploading. System intentionally rejects empty scans. |
| **`sqlite3.OperationalError: database is locked`** | Long-running transaction or non-WAL mode | Check `PRAGMA journal_mode;` in SQLite. Ensure WAL mode is active. |
| **`ZeroDivisionError` in calculator** | Unsafe arithmetic query submitted | Handled gracefully by AST calculator; returns structured error without crashing. |
