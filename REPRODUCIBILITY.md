# Reproducing Labs 1-7

## Supported environment

Windows 11 x64, PowerShell, Python **3.13**, and one isolated `.venv` per lab. `python` need not be on PATH: setup discovers the interpreter through `py -3.13`, or accepts `-Python <executable>`. Direct requirements and complete Windows/Python 3.13 transitive locks are committed for every lab. Do not combine the lab environments: LangChain, agents, notebooks and model stacks have different constraints.

```powershell
.\setup.ps1 -Lab 1
.\run.ps1 -Lab 1 -Smoke
.\run.ps1 -Lab 1
```

Repeat with the required lab number. Labs 1, 3-6 run their preserved notebooks in order in one private kernel; this also supplies state shared by consecutive Max notebooks. An executed copy, including partial output on failure, goes to the ignored `outputs/reproducibility-run.ipynb`. Lab 2 opens its CLI. Lab 7 opens Streamlit. Notebook experimental matrices can take substantially longer than the short functional tests.

## NVIDIA RTX 3050 4 GB

For Labs 4-5 and the Max studio, use `setup.ps1 -Lab <number> -Cuda`. This selects **torch 2.14.1+cu130** from `https://download.pytorch.org/whl/cu130`; other dependencies remain locked. It requires a driver compatible with CUDA 13.0. The wheel supplies its runtime; a separate CUDA toolkit is not required. The verification machine reports driver **610.88**. CPU environments use torch 2.14.1+cpu. NNTS Lab 7 deliberately embeds on CPU.

Run batch size 1 at 512x512. SD uses FP16 and CPU offload on CUDA. The Max studio unloads the other vision model before switching and serializes complete inference calls across Streamlit sessions. Its Ollama model runs on CPU. Do not run a second GPU workload concurrently. The 16 GB RAM verification machine also experienced paging when both LM Studio and Ollama were loaded alongside multiple model processes; heavy verification was serialized. CPU-only vision performance and different driver/Python combinations are not covered by the GPU evidence.

## Model and input integrity

`models.lock.json` records immutable Hugging Face revisions and upstream SHA-256 values for large model files. `scripts/fetch_models.py` downloads the selected revision and verifies every downloaded large file against that record. Normal model downloads are part of setup. Downloads require network access; no file from an earlier project directory is required. Existing standard Hugging Face caches may be reused after checksum verification.

```powershell
# Use the selected lab interpreter, for example:
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/fetch_models.py llava
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/fetch_models.py diffusion --fp16
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/fetch_models.py minilm
```

The original LLaVA 0.5B and SD 1.5 checkpoints are used, not tiny substitute models. The model downloader verifies provenance, not semantic correctness or bit-identical GPU output. Seeds, versions and source revisions establish the reproducible configuration; GPU kernels can still differ between hardware/runtime combinations.

`inputs.sha256.json` records the valuable inputs already distributed in Git. Text checksums normalize CRLF to LF; binaries are checked byte-for-byte. Experimental notebooks, examples, gallery samples, datasets and benchmark outputs are preserved. Virtual environments, model caches, smoke images/databases and executed notebook artifacts are ignored.

## Service provisioning

Service identities/digests are recorded in `services.lock.json`. Never expose the unauthenticated local endpoints publicly.

Start Docker Desktop's Linux engine. Lab 7 setup pulls Redis Stack 7.4.0-v8 and checks its image digest. Lab 3 uses port 6379; Lab 7 uses port 6380. Start Lab 3 Redis with `docker compose -f labs/lab03_rag/docker-compose.yml up -d`. These Compose volumes are persistent and are not removed by run/setup.

Install LM Studio and start its local API at http://localhost:1234/v1. Load `mistralai/ministral-3-3b` with **16384 context tokens or more**, GPU offload disabled. This exact model was used in live tests. For a reproducible GGUF download, run the Lab 7 interpreter with `scripts/fetch_models.py ministral`; import the resulting Q4_K_M GGUF into LM Studio. The downloader prints its cache path and verifies the pinned upstream SHA-256. The GGUF source is `lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF`, revision recorded in `models.lock.json`. Loading a model and enabling the server are explicit LM Studio configuration steps. No cloud key is needed. `LM_STUDIO_MODEL` and `LM_STUDIO_BASE_URL` override the defaults.

NNTS datasets are reconstructed by `scripts/fetch_data.py` during Lab 1/3 setup. `datasets.lock.json` records versioned Kaggle archive URLs, archive SHA-256, extracted file SHA-256 and the SMS schema conversion. Sources: SMS Spam Collection (`uciml/sms-spam-collection-dataset`, version 1), BBC (`yufengdev/bbc-fulltext-and-category`, version 2), and Shopping (`anvitkumar/shopping-dataset`, version 611). The three Shopping CSV hashes exactly match the existing lab manifest. If a source requires access approval or an archive changes, setup fails rather than silently using different data. Respect each source's published terms.

Lab 7 standalone: `labs/lab07_final_project/setup.ps1`, then `labs/lab07_final_project/run.ps1`. Models/images are provisioned during setup, not routine startup. Default UI port 8501; use `-Port 8503` on the lab-local launcher if 8501 is owned by another application. Configure both Redis index and prefix to isolate a second checkout. See the lab README for uploads, sessions, restart tests and optional REST API.

## Verification scope

`run.ps1 -Lab N -Smoke` runs real functionality and fails when a required service/model is unavailable. Lab 1 takes forward/backward/optimizer steps through both real model classes; Max also reads its distributed Enron data. Lab 2 calls the actual client implementation. Lab 3 builds/searches real Redis or Chroma vectors and calls the local LLM. Lab 4 generates a real 512x512 image using the original SD checkpoint; eight diffusion steps and a nonblack-output check test execution, not output quality. Lab 5 performs real visual inference. Lab 6 requires an actual agent tool call. Lab 7 tests real model/service workflows, SQLite/image persistence and Streamlit rendering.

These smoke tests do not retrain complete historical experiments or revalidate every experimental metric. Historical notebook outputs and timings remain labeled as historical. Passing imports or injected regression responses are not counted as live inference. See `VERIFICATION.md` for the exact executed record and limitations.

## Updating locks

After intentionally changing direct requirements, resolve each lab separately with `uv pip compile <lab>/requirements.txt --python-version 3.13 --python-platform windows --output-file <lab>/requirements.lock.txt`. Install in a clean lab environment, run the live smoke, then commit both requirement files. `uv` is a developer convenience; the PowerShell setup scripts use standard Python/pip. Do not regenerate model revisions/checksums implicitly on application startup.

## Capstone UI and restart checks

With the prepared Lab 7 interpreter and LM Studio running:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/smoke_copilot_ui.py
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/smoke_redis_restart.py
.\labs\lab07_final_project\.venv\Scripts\python.exe scripts/smoke_launcher.py
```

The UI check submits a real Streamlit chat message and checks the saved SQLite turn. The Redis check creates its own labeled container/volume on a reserved loopback port, uploads Markdown/TXT/PDF, restarts Redis, verifies index/key persistence and queries the uploaded document. It removes only its own container and volume. The launcher check restarts the real PowerShell launcher on port 8503 and verifies SQLite persistence and owned-child cleanup.
