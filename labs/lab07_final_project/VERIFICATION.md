# Startup verification — 2026-11-13

Executed on Windows 11, PowerShell, Python 3.13.5, Docker Engine 29.7.2,
an already running `lab07-redis`, and LM Studio with `mistralai/ministral-3-3b`.
No models or images were downloaded. Existing environments and persistent data
were retained. All source changes are confined to Lab 7.

| Requested scenario | Result |
| --- | --- |
| Launch from Lab 7 | Executed `run.ps1 -NoBrowser`; reached Application ready and HTTP health `ok`. |
| Absolute path from another directory | Executed PowerShell `-File <absolute run.ps1> -NoBrowser` with the Windows temporary directory as CWD; reached Application ready. |
| PyCharm terminal | Not executed in the PyCharm UI. PowerShell behavior and the configured project interpreter were tested outside it. |
| Redis already running | Live startup reused it. Container remained running after supervisor shutdown. |
| Docker unavailable | Fault-injection test verified missing CLI diagnostic before services start. Docker Desktop was not actually stopped. |
| LM Studio unavailable | HTTP connection error injected into the real readiness probe; SERVER_UNREACHABLE verified. The user's server was not stopped. |
| No LLM loaded | Empty `/models` response injected into the real readiness probe; NO_MODEL_LOADED verified. Also tested wrong model ID and HTTP 200 without tool calls. |
| Streamlit port occupied | Live TCP listener on 8501; launcher rejected it with the precise port diagnostic and did not terminate the listener. |
| Repeated startup | Live second launcher rejected with the already-running diagnostic. Subsequent fresh startup also reached readiness. |
| Clean shutdown | Sent Windows Ctrl+Break to a dedicated launch process group after readiness. Python printed the persistent-data retention message; harness observed exit 0. No Lab 7 Python processes remained. Interactive Ctrl+C in PyCharm remains unverified. |
| Abrupt supervisor termination | Terminated only the identified test supervisor; its Windows Job Object removed the Streamlit tree. A separate unit test also exercised a real child process. Redis remained running. |
| Retrieval and LLM response | Live hybrid search returned five passages. Two real ADR 001 questions each produced GROUNDED answers, six evidence passages, and real citation IDs. |
| UI initialization / imports | Streamlit AppTest loaded the canonical UI without exceptions and verified Standalone In-Process as the default. Editable installation, compilation, and imports succeeded without PYTHONPATH changes. |

Automated suite: **12 tests passed** with
`python -m unittest discover -s tests -v`. Tests cover readiness polling, early
child exit, occupied ports, missing Docker, model protocol failures, ownership
of Redis cleanup, interrupt cleanup, and Windows process-tree containment.
Fault-injection tests are not live end-to-end tests.

`pip install --dry-run --no-index -r requirements.lock.txt` succeeded against the
existing environment. Python metadata was checked for 3.12 compatibility, but a
fresh installation on Python 3.12 or 3.13 was not performed. The legacy environment
inherits another environment's packages; `pip check` reports an unrelated existing
conflict: `llama-index-llms-openai 0.8.3` requires `openai<3`, while Lab 7 uses
`openai==3.17.0`. The launcher checks Lab 7's dependency snapshot; this unrelated
package is not used by Lab 7. Setup reports remaining conflicts.

Not verified live: automatic browser opening (`-NoBrowser` was used), launching
Docker from a stopped state, cleanup of a newly started Redis container (covered
by fault injection), the optional standalone FastAPI server, PDF upload, and
interactive PyCharm terminal behavior. Browser-free AppTest verified rendering
and initialization, not a browser-driven chat submission.

`compileall` and `git diff --check` passed. Runtime scripts, logs, traces, databases,
and diagnostic snapshots remain under ignored `outputs/`; none are deliverables.
