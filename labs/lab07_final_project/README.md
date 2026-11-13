# Lab 07 — Local Engineering Knowledge Copilot

Local retrieval with Redis Stack, CPU embeddings, LM Studio tool calling, SQLite
session persistence, and Streamlit. The UI uses services directly; FastAPI is optional.

## First installation — Windows 11

Prerequisites:

- 64-bit Python **3.12 or 3.13**, PowerShell, Docker Desktop with its Linux engine running, and Compose v2.
- Redis Stack image: explicitly run `docker pull redis/redis-stack:7.4.0-v8` once.
- LM Studio: load a tool-capable chat model (tested with `mistralai/ministral-3-3b`) and start its local server at `http://localhost:1234/v1`.
- Cached embedding model `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

From this directory:

```powershell
.\setup.ps1 -Python 'C:\Path\To\Python313\python.exe' -DownloadEmbedding
```

`-DownloadEmbedding` explicitly permits the initial embedding download. Omit it
if already cached. Setup reuses the Lab 7 `.venv`; new environments are isolated
from other labs. It installs `requirements.lock.txt` and the editable package.
Direct requirements are in `requirements.txt`. The dependency snapshot was resolved
on Windows/Python 3.13; a fresh Python 3.12 installation remains unverified.
Matching existing CUDA torch builds satisfy the public version pin.

Optional: copy `.env.example` to `.env` and edit it. Shell variables take precedence.
No API key is needed. Relative `SESSION_DB` paths resolve under Lab 7.
Startup never installs packages, pulls images, or downloads models.

## Start and stop

```powershell
.\run.ps1
```

This also works in PyCharm's PowerShell terminal. Select `.venv\Scripts\python.exe`
as its interpreter. An absolute path works from another directory:

```powershell
& 'C:\Path\To\Repository\labs\lab07_final_project\run.ps1'
```

Startup validates dependencies, Docker, Redis Search, inference, actual function
tool calls, corpus integrity, and indexing. It opens [the UI](http://localhost:8501)
after Streamlit's health endpoint responds. The UI defaults to **Standalone In-Process**.
No separate API is started or advertised. `-NoBrowser` suppresses browser opening;
`-Python <venv interpreter>` selects another prepared project environment.

Keep the terminal open. Press **Ctrl+C** to stop. Only the launcher's Streamlit
process tree and a Redis container started by this invocation are stopped.
Previously running Redis is retained. Volumes and SQLite files are never removed.
A Windows Job Object kills the Streamlit tree if its supervisor dies abruptly;
Docker may remain running after abrupt termination. A second launcher is rejected.

The eight default documents are in `assets/knowledge/`. Content hashes avoid
re-embedding unchanged documents; missing Redis vectors are restored from SQLite.
Initial ingestion failures block startup. Uploads and history persist in Redis's
Compose volume and `outputs/sessions/`. Logs are in `outputs/startup.log`.

## Troubleshooting

| Error | Resolution |
| --- | --- |
| Missing environment / incompatible Python | Run setup with a 64-bit Python 3.12–3.13 executable. |
| Missing dependency / `No module named knowledge_copilot` | Run `.\setup.ps1`, including editable installation. No `PYTHONPATH` changes are needed. |
| Docker unavailable | Start Docker Desktop; confirm `docker info` succeeds. |
| Redis image missing | Run the explicit `docker pull` command above. |
| Redis not ready / port 6380 occupied | Check `docker compose logs redis` and `Get-NetTCPConnection -LocalPort 6380`. Resolve the conflict yourself; startup uses only the configured endpoint. |
| LM Studio unreachable | Start its local server and check `http://localhost:1234/v1/models`. |
| No model / wrong model / unsupported tools | Load a tool-capable chat model. Set `LM_STUDIO_MODEL` to its exact API ID, or leave it unset for discovery. |
| Missing embedding cache | Run `.\setup.ps1 -DownloadEmbedding` with network access. Custom models need their own cached revision. |
| Port 8501 occupied / already running | Use the existing UI or stop its launcher. Inspect `Get-NetTCPConnection -LocalPort 8501`; startup does not kill the owner. |
| UI fails after server readiness | Inspect `outputs/startup.log`. UI initialization reports service/corpus errors on screen. |
| Legacy environment dependency conflicts | Inspect `pip check`. Use a clean Lab 7 environment if inherited unrelated packages conflict; other labs' environments are not modified. |
| PowerShell blocks scripts | Follow your organization's execution policy. The scripts do not change system policy. |

## Optional REST API

For API development, use the same shared initialization:

```powershell
.\.venv\Scripts\python.exe -m uvicorn knowledge_copilot.api.app:app --host 127.0.0.1 --port 8000
```

[API docs](http://localhost:8000/docs) are available only while this process runs.
Routes include `/api/v1/health/live`, `/api/v1/health/ready`, `/api/v1/documents`,
`/api/v1/chat`, and `/metrics`. The UI's optional HTTP mode requires this server.

## Verification and implementation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [VERIFICATION.md](VERIFICATION.md) for executed scenarios and limitations.
Canonical code is in `src/knowledge_copilot/`; imports use the installed
`knowledge_copilot` package. The notebook remains at
`notebooks/01_final_project.ipynb`; compatibility entrypoints `app.py` and
`web_app.py` remain available.
