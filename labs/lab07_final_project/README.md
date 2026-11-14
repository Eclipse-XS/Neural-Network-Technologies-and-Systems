# Lab 7: Local Engineering Knowledge Copilot

Redis Stack retrieval, CPU E5 embeddings, LM Studio tool calling, Streamlit, and SQLite document/session persistence. Canonical code is in `src/knowledge_copilot`; the editable installation avoids PYTHONPATH changes. The UI calls services directly. FastAPI is optional.

## Fresh installation on Windows 11

Install 64-bit Python 3.13, Docker Desktop with its Linux engine running, and LM Studio. Python 3.12 is accepted by the package but has not been validated with this dependency lock.

Load `mistralai/ministral-3-3b` in LM Studio with context **16384 or larger**, GPU offload **disabled**, and start its local server on port 1234. A 4096-token model passed the simple readiness probe but failed the actual RAG workflow. The exact GGUF source revision and checksum are in the root model lock; see the root reproduction guide for download instructions.

From this directory:

```powershell
.\setup.ps1
.\run.ps1 -NoBrowser
```

Setup creates `.venv`, installs `requirements.lock.txt` and the editable package, downloads and verifies the pinned E5 weights, pulls Redis Stack and checks its image digest. `pip check` failures abort setup. Use `-Python <executable>` to select Python 3.13. `-SkipEmbedding` and `-SkipDocker` skip provisioning for deliberate component testing. The old `-DownloadEmbedding` option remains accepted for compatibility; downloads are now the default.

Copy `.env.example` to `.env` only if overriding defaults. Shell variables take precedence. No cloud API key is needed. `SESSION_DB` resolves under the lab directory. Set both `REDIS_INDEX_NAME` and `REDIS_KEY_PREFIX` when isolating another application instance from an existing index.

## Startup, upload and persistence

`run.ps1` validates Python/dependencies, Docker, Redis Search, real LM Studio inference/tool calling, local embedding weights and corpus integrity before opening Streamlit. Routine startup does not install dependencies or download models. The default URL is http://localhost:8501. Use `-Port 8503` when another application owns 8501; the launcher never terminates the port owner. An absolute launcher path works from another terminal directory.

Use the document management tab to upload Markdown, text or PDF; the application extracts, chunks and indexes the content. Ask a document-specific question in chat and inspect the returned citations. The bundled corpus is in `assets/knowledge`. Content hashes prevent duplicate embeddings; missing vectors can be restored from SQLite. Uploaded document metadata/chunks, sessions and canonical turns persist in `outputs/sessions/assistant_memory.sqlite`; Redis persists in its Compose volume.

Keep the terminal open. Ctrl+C stops owned Streamlit processes and Redis only if this invocation started it. Previously running Redis, its volume and SQLite are retained. The Windows Job Object also removes the owned Streamlit process tree if the supervisor dies abruptly. A duplicate launcher is rejected before importing heavyweight services. Startup logs are under `outputs/startup.log`.

## Verification

From the repository root:

```powershell
.\run.ps1 -Lab 7 -Smoke
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests -q
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/smoke_launcher.py
```

The live smoke uses a unique Redis index/prefix and temporary SQLite database, uploads a document, asks the real model a question, checks its answer/citations, reopens SQLite, reindexes/deletes the document, and renders the real Streamlit UI. The launcher test starts and restarts `run.ps1` on port 8503, checks duplicate rejection, SQLite persistence and owned-child cleanup. It deliberately refuses an already occupied test port. The regression suite includes fault injection; those cases are not evidence of live service execution. See [VERIFICATION.md](VERIFICATION.md) for the current record.

## Optional REST API

```powershell
.\.venv\Scripts\python.exe -m uvicorn knowledge_copilot.api.app:app --host 127.0.0.1 --port 8000
```

The API exposes `/api/v1/documents`, `/api/v1/chat`, session retrieval, health endpoints and `/metrics`. API documentation is available at http://localhost:8000/docs only while this server runs. The UI HTTP mode requires this separate server; standalone UI mode does not.

## Diagnostics

- Docker unavailable: start Docker Desktop and verify `docker info`.
- Missing model/cache/dependency: rerun setup with network access. Do not enable synthetic embedding fallback for application use.
- LM Studio context overflow: reload the model with at least 16384 context tokens.
- Unsupported tool calling/wrong ID: use the exact model ID reported by `/v1/models` and a tool-capable checkpoint.
- Redis port 6380 or UI port occupied: inspect the owning application; configure an unused endpoint/port.
- Errors after HTTP readiness: inspect `outputs/startup.log` and UI errors. An HTTP health response alone does not establish inference success.
