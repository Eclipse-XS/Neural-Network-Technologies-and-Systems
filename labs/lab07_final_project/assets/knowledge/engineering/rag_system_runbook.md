# Operational Runbook: Local Engineering Knowledge Copilot

## 1. System Overview
The copilot runs locally as a modular monolith. It consists of:
1. FastAPI backend server running on port 8000 (`src/knowledge_copilot/api/app.py`).
2. Streamlit interactive UI running on port 8501 (`src/knowledge_copilot/ui/streamlit_app.py`).
3. Redis Stack container running on port 6380 for vector indexing.
4. LM Studio local inference server running on port 1234 serving Ministral-3B.

## 2. Standard Operating Procedures (SOP)

### Starting Local Services
1. Start the Redis Stack container:
   `docker start lab07-redis` (or `docker compose up -d redis`)
2. Verify Redis connectivity:
   `redis-cli -p 6380 ping` (expected output: `PONG`)
3. Launch LM Studio and start the local server on `http://127.0.0.1:1234/v1` with `mistralai/ministral-3-3b`.
4. Start the API server:
   `python -m uvicorn labs.lab07_final_project.src.knowledge_copilot.api.app:app --host 127.0.0.1 --port 8000 --reload`
5. Start the Streamlit application:
   `streamlit run labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py --server.port 8501`

### Reindexing and Disaster Recovery
- If vector embeddings become desynchronized or corrupted:
  1. Trigger full reindexing via CLI:
     `python -m labs.lab07_final_project.src.experiments --reindex`
  2. Or purge the Redis vector index:
     `redis-cli -p 6380 EVAL "for i, name in ipairs(redis.call('KEYS', 'lab07:kb:*')) do redis.call('DEL', name) end" 0`
  3. The next API startup or query will trigger clean deterministic chunk ingestion.

### Health Probes & Monitoring
- Ingress readiness probe: `curl -s http://127.0.0.1:8000/api/v1/health/ready`
- Expected response: `{"status":"ready","database":true,"redis":true}` with HTTP 200.
- If Redis fails, the system transitions to degraded state (BM25 lexical search only).
