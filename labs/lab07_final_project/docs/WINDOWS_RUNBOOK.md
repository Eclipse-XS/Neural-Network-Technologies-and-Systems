# Windows Step-by-Step Runbook & Demonstration Guide

This runbook contains verified, exact commands for running, testing, demonstrating, and evaluating Lab 7 on Windows 11 with WSL2 Redis and local LM Studio.

---

## 1. Environment & Prerequisites

- **OS**: Windows 11 (64-bit)
- **WSL2**: Ubuntu instance hosting Redis Stack 8.10.3 (Search module enabled)
- **Python**: 3.13.5 in `labs/lab07_final_project/.venv`
- **PyTorch**: 2.14.1+cu130 (NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM)
- **LM Studio**: Listening on `http://localhost:1234/v1` with loaded model `mistralai/ministral-3-3b`
- **Embedding model**: `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, 384 dim, 512 max seq len, CPU inference)

---

## 2. Step-by-Step Operations

### Step 1: Open Terminal & Navigate to Project Root
```powershell
cd C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems
```

### Step 2: Activate the Lab 7 Python Virtual Environment
```powershell
& "labs/lab07_final_project/.venv/Scripts/Activate.ps1"
# Verify Python location
where.exe python
```

### Step 3: Verify CUDA & GPU Acceleration
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
Expected output:
```text
PyTorch: 2.14.1+cu130 | CUDA available: True | GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

### Step 4: Verify Redis Service in WSL2 (Port 6380)
If Redis is not running in WSL2, start it:
```powershell
wsl -d Ubuntu -e redis-server --port 6380 --loadmodule /opt/redis-stack/lib/redisearch.so --daemonize yes
```
Verify connectivity from Windows:
```powershell
python -c "import redis; r = redis.Redis(host='localhost', port=6380); print('Redis version:', r.info('server')['redis_version'])"
```
Expected output:
```text
Redis version: 8.10.3
```

### Step 5: Verify Redis Search Module
```powershell
python -c "import redis; r = redis.Redis(host='localhost', port=6380); print('Loaded modules:', [m[b'name'].decode('utf-8') for m in r.module_list()])"
```
Expected output:
```text
Loaded modules: ['search', 'vectorset']
```

### Step 6: Verify LM Studio Local LLM Server (Port 1234)
Launch LM Studio, load `mistralai/ministral-3-3b`, and start the Local Server on port 1234.
Verify via HTTP probe:
```powershell
python -c "import httpx; res = httpx.get('http://localhost:1234/v1/models', timeout=5.0); print('LM Studio Status:', res.status_code, '| Models:', [m['id'] for m in res.json()['data']])"
```
Expected output:
```text
LM Studio Status: 200 | Models: ['mistralai/ministral-3-3b', ...]
```

### Step 7: Check Dependencies & Package Integrity
```powershell
python -m pip check
```
Expected output:
```text
No broken requirements found.
```

### Step 8: Build / Verify Knowledge Base Vector Index
```powershell
python -c "from labs.lab07_final_project.src.config import Settings; from labs.lab07_final_project.src.knowledge_base import KnowledgeBase; kb = KnowledgeBase(Settings()); print('Index Status:', kb.initialize())"
```
Expected output:
```text
Index Status: {'index_name': 'lab07_course_kb_idx', 'key_prefix': 'lab07:kb', 'indexed_count': 112, 'embedding_model': 'intfloat/multilingual-e5-small', ...}
```

### Step 9: Run Held-Out Retrieval Evaluation (30 Queries, 3 Modes)
```powershell
python labs/lab07_final_project/experiments/run_held_out_eval.py
```
Outputs saved to `labs/lab07_final_project/artifacts/portfolio/held_out_eval/`:
- `hybrid_metrics.json`
- `dense_metrics.json`
- `lexical_metrics.json`

### Step 10: Run Automated Unit and Integration Tests (Labs 5–7)
```powershell
python -m pytest labs/lab05_multimodal/tests labs/lab06_agents/tests labs/lab07_final_project/tests -q
```
Expected output:
```text
73 passed, 1 skipped in ~70s
```

### Step 11: Run Offline Replay Notebook Check
```powershell
python -m pytest labs/lab07_final_project/tests/test_notebook.py -v
```
Expected output:
```text
PASSED in ~3s
```

### Step 12: Launch Interactive CLI Assistant
```powershell
python labs/lab07_final_project/app.py
```
Ask:
1. `What model was used in Lab 6?`
2. `According to the reports, how many outputs were generated in Lab 4 and how many VQA inferences in Lab 5?`
Type `exit` or `quit` to end the session.

### Step 13: Launch Streamlit Web UI
```powershell
streamlit run labs/lab07_final_project/web_app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

### Step 14: Demonstrate Three Key Web Scenarios

1. **Exact Grounded Fact (RAG)**:
   - Question: `Which Stable Diffusion model was used in Lab 4?`
   - Expected answer: `runwayml/stable-diffusion-v1-5` (revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`)
   - Click `Tool trace and evidence` expander: verify `search_knowledge` tool call and citation `[lab04#chunk-021f33e4ebbbb4ea]`.

2. **Cross-Document Arithmetic (Tools + RAG)**:
   - Question: `How many generated outputs were produced in Lab 4 and how many VQA generations were performed in Lab 5? Which lab produced more, and by how many?`
   - Expected answer: Lab 4 = 13, Lab 5 = 17, difference = 4 (17 - 13).
   - Tool trace: verify retrieved operands from `lab04` and `lab05`.

3. **Memory & Context Tracking**:
   - Turn 1: `What model was used in Lab 6?`
     - Response: `mistralai/ministral-3-3b`
   - Turn 2: `What main behavioral weakness was observed with it?`
     - Response: Resolves `it` to `mistralai/ministral-3-3b` from conversation history and retrieves context limit or tool selection observations.
   - Click `New session` in the sidebar and repeat Turn 2:
     - Response: Deterministic ambiguity guard asks for clarification because the antecedent does not exist in a new session.

### Step 15: Safe Clean Shutdown
- Stop Streamlit: `Ctrl + C` in the Streamlit terminal.
- Do NOT kill or modify port 8000 (reserved for the user's independent project).
- To stop WSL Redis if desired:
```powershell
wsl -d Ubuntu -e redis-cli -p 6380 shutdown
```
