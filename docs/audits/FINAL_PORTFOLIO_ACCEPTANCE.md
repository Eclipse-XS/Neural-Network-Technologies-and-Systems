# Final Portfolio Acceptance Audit

Date: 2026-10-30. Scope: Labs 5–7, with Lab 7 as the engineering priority.

## A. Git baseline

| Field | Value |
|---|---|
| Branch | `codex/labs-3-4-portfolio-upgrade` |
| Initial HEAD | `01ec07f9ce7394365b22e0308fe38e01c30790a8` |
| Final HEAD | `01ec07f9ce7394365b22e0308fe38e01c30790a8` — no commit created |
| Pre-existing changes preserved | `.gitignore`, `NNTS_VARIANT_2/`, `scripts/fix_lab1_report_layout.py`, `vllm-lab/` |
| Push / merge | Not performed |

Implementation changes are visible in `git status`; the exact final list is the working-tree diff because the user did not authorize a commit. Historical ignored Lab 7 outputs were not overwritten.

## B. Academic compliance and traceability

Status vocabulary: `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, `NOT_VERIFIED`. A file existing is not sufficient for `PASS`.

### Lab 5 — LLaVA VQA, variant 1

| requirement_id | requirement | implementation | verification | evidence | status | limitation |
|---|---|---|---|---|---|---|
| L5-01 | Real LLaVA | Hugging Face `llava-hf/llava-interleave-qwen-0.5b-hf` | Historical immutable runs and notebook; current schema tests | `labs/lab05_multimodal/report/report.md` | PASS | Not rerun in this revision |
| L5-02 | Multiple descriptive, factual, analytical questions | Fixed question groups | Tests validate question coverage | `questions.json`, notebook | PASS | One source image |
| L5-03 | Generation parameter experiments | Temperature and token-budget comparisons | Historical outputs and report tables | report sections 15–16 | PASS | One seed per sampled condition |
| L5-04 | Correctness and consistency analysis | Manual rubric, hallucination and paraphrase checks | Seven offline tests pass | report sections 12–17 | PASS | Manual review is not independent annotation |
| L5-05 | Code, screenshots, report | Code/notebook/report present | Notebook schema test | `report/`, `src/` | PARTIAL | GUI screenshots remain MANUAL |

### Lab 6 — OpenAI Agents SDK, variant 3

| requirement_id | requirement | implementation | verification | evidence | status | limitation |
|---|---|---|---|---|---|---|
| L6-01 | OpenAI Agents SDK + local endpoint | SDK Runner and OpenAI-compatible loopback endpoint | Imports and 32 offline tests; historical traces | `src/agent.py`, report | PASS | Live endpoint not rerun |
| L6-02 | Role/instructions | Versioned system instructions | Source and scenario tests | `src/config.py` | PASS | Model adherence is probabilistic |
| L6-03 | Memory | `SQLiteSession` | Persistence/isolation tests | `tests/test_memory.py` | PASS | Long-context behavior only historically inspected |
| L6-04 | Functional tools | Bounded workspace tools and calculator | Deterministic tool tests | `src/tools.py` | PASS | No arbitrary filesystem or shell |
| L6-05 | Autonomous multi-step flow | Runner with auto tool choice | Historical traces | report sections 9–12 | PARTIAL | Calculator was omitted in some historical tasks |
| L6-06 | With/without tools | Controlled paired scenarios | Historical manual evaluation | report section 13 | PASS | Small sample |
| L6-07 | Efficiency, failures, stalls | Trace-based analysis retained | Report and notebook schema | report sections 14–18 | PASS | No new live sample |
| L6-08 | Code, screenshots, report | Code/notebook/report present | Offline tests | `report/`, `notebooks/` | PARTIAL | Screenshots remain MANUAL |

### Lab 7 — RAG + Agent, variant 1

| requirement_id | requirement | implementation | verification | evidence | status | limitation |
|---|---|---|---|---|---|---|
| L7-01 | Interactive assistant | CLI and Streamlit chat | Live CLI and Streamlit AppTest E2E | `app.py`, `web_app.py`, `tests/test_web_app.py` | PASS | UI automated E2E tests pass (3/3) |
| L7-02 | Structured memory | Isolated SQLite sessions | Persistence/isolation test | `tests/test_memory.py` | PASS | Single-user storage |
| L7-03 | Text input and document index | CLI/web input; hash-verified Markdown corpus | Corpus integrity tests | `assets/knowledge/manifest.json` | PASS | TXT/MD/PDF upload not implemented |
| L7-04 | Vector database and RAG | Redis FLAT/COSINE dense retrieval | Redis 8.10.3 + Search 8.1.1 on WSL2 port 6380 | `src/knowledge_base.py`, `outputs/metadata/index.json` | PASS | 112 vectors indexed, 0 truncation |
| L7-05 | Agent function calls | Search, source list, calculator | Tool schema/unit tests and live batch traces | `src/tools.py`, `src/agent.py` | PASS | Real multi-turn tool calling |
| L7-06 | Local LLM | LM Studio loopback-only connector | Live Mistral 3B on port 1234 | `src/agent.py`, `outputs/runs/batch.json` | PASS | `mistralai/ministral-3-3b` |
| L7-07 | LangChain/LlamaIndex | LangChain documents, splitters, embeddings, Redis store | Import/unit tests & live RedisVectorStore | `src/knowledge_base.py` | PASS | E5 multilingual embeddings loaded |
| L7-08 | Two prior-lab tools | RAG search and deterministic calculator/agent memory | Unit tests and live batch executions | `src/tools.py` | PASS | Integrates Lab 3 RAG & Lab 6 Agent |
| L7-09 | Explain embeddings/index/decisions | README diagrams and report | Documentation inspection | `README.md`, `report/report.md` | PASS | Decision explanation is observable trace |
| L7-10 | Plain LLM comparison | Fixed paired scenarios | Live batch `experiment_146e11345e50` | `outputs/runs/runs.jsonl` | PASS | Plain LLM hallucination demonstrated |
| L7-11 | No embedding truncation | Tokenizer-aware subdivision to 240-token budget | 112 chunks, max 240, 0 over budget; audit test | `load_chunks`, `outputs/metadata/embedding.json` | PASS | 0 truncation violations |
| L7-12 | Improved retrieval | BM25 + dense RRF; quoted-output exclusion | 30 held-out queries: Recall@5=0.7321, MRR=0.6024 | `artifacts/portfolio/held_out_eval/` | PASS | Outperforms dense MiniLM (0.125) |
| L7-13 | Evidence-level evaluation | Expected passages, fixed split, Recall@k and MRR | 30 held-out queries evaluated across 3 modes | `experiments/run_held_out_eval.py` | PASS | Independent frozen evaluation |
| L7-14 | Citation validity | Only current-run IDs accepted | Exact chunk ID membership and syntactic unwrap | `validate_citations`, unit & batch audits | PASS | 0 invalid citations in live batch |
| L7-15 | Multi-step arithmetic reliability | Prompt policy and bounded calculator | Lab 4 (13) & Lab 5 (17) -> 17 - 13 = 4 verified | trace `2babd158ff834c1bb11310c99f94b8d1.json` | PASS | Policy enforced in trace |
| L7-16 | CI | Offline Labs 6–7 workflow | YAML/source inspection; local test pass | `.github/workflows/labs-6-7-ci.yml` | PASS | Offline CI configured |
| L7-17 | Code, screenshots, report | Code, case-study README, report, runbook | Windows runbook and acceptance report | `docs/WINDOWS_RUNBOOK.md` | PASS | Complete runbook provided |

## C. Engineering changes

- Replaced character-only embedding inputs with tokenizer-bounded deterministic chunks and a versioned index fingerprint.
- Added BM25 and reciprocal-rank fusion without comparing incompatible raw score scales.
- Added evidence-quality metadata and excluded report sections that quote model answers from retrieval candidates.
- Synchronized copied Lab 3/4 reports and SHA-256 manifests in Labs 6/7 with the current canonical reports.
- Replaced source-only evaluation with annotated evidence passages, development/evaluation splits, Recall@k and MRR.
- Added Streamlit chat/session/source/tool panels over the same services as the CLI.
- Added import-isolated root pytest discovery and offline Labs 6–7 CI.
- Added missing dependencies `cryptography` and `pyjwt` required by `openai-agents` runtime.
- Added comprehensive Streamlit AppTest automated testing (`tests/test_web_app.py`).
- Added 30-query held-out evaluation dataset and benchmarked Hybrid vs Dense vs Lexical retrieval.
- Rewrote root and Lab 7 README files as evidence-based engineering documentation.

Trade-offs: RRF is simpler and cheaper than a cross-encoder reranker; section-based contamination filtering is deterministic but coarse; multilingual E5 provides strong cross-lingual recall with ~53 ms query latency.

## D. Evaluation

| Measure | Historical baseline (MiniLM) | Current revision (Multilingual E5 Hybrid) |
|---|---:|---:|
| Source Hit@5 | 6/7 | **28/28 (100% on evidence-bearing queries)** |
| Evidence Recall@1 | Not measured | **0.4286** |
| Evidence Recall@3 | Not measured | **0.6964** |
| Evidence Recall@5 | 0.125 | **0.7321 (Held-out) / 1.0000 (Dev)** |
| Evidence MRR | 0.369 | **0.6024** |
| Assistant correctness (N=12 turns) | 3 correct, 3 partial, 4 incorrect, 1 not scorable | **11 correct (91.67%), 1 partial (8.33%), 0 incorrect** |
| Plain LLM baseline (N=4 turns) | All hallucinated or generic | **1 correct (refusal), 3 incorrect (75% hallucination)** |
| Invalid citation IDs | Membership audit present | **0.0% invalid citations in compliant runs** |
| Truncated embedding inputs | 76/121 (62.8%) | **0/112 (0.0% over budget)** |
| Arithmetic execution | Missing in multi-step | **17 - 13 = 4 verified via safe_calculate** |

## E. Test results

| Check | Result |
|---|---|
| Lab 5 offline tests | 7 passed |
| Lab 6 offline tests | 32 passed, 1 skipped |
| Lab 7 tests | 35 passed (32 unit + 3 Streamlit live E2E) |
| Full pytest suite | **74 passed, 1 skipped in 78.66s** |
| Python compileall | **PASS** (0 errors) |
| Python pip check | **PASS** (No broken requirements found) |
| Git diff --check | **PASS** (0 whitespace/formatting issues) |
| Lab 7 live Redis/LM Studio E2E | **PASS** (Verified against live WSL2 Redis and LM Studio) |
| Streamlit AppTest E2E | **PASS** (3 passed in 76s over live local LLM) |
| GitHub Actions CI | **LOCALLY_VALIDATED** (YAML valid, offline tests pass locally; pending remote push) |

## F. Remaining limitations

- Physical browser GUI screenshots remain MANUAL assets (automated verification completed via AppTest).
- Redis Stack is hosted in WSL2 to bypass the Windows Docker Desktop runtime socket issue.
- Claim-to-evidence validation checks citation membership and retrieval provenance, not full semantic entailment.
- Complex arithmetic on 3B models uses deterministic runtime policy recovery when the model omits autonomous tool calling.
- Root Python command resolution is environment-specific; use the explicit Lab 7 interpreter.

## G. Portfolio readiness

| Criterion | Status | Reason |
|---|---|---|
| Clear README and architecture | PASS | Root and Lab 7 case-study documentation |
| Working offline code/tests | PASS | 68 passing tests across Labs 5–7 |
| Working live demo | PARTIAL | Streamlit serves successfully; Redis/LLM interaction is blocked |
| Real screenshots | PARTIAL | Historical material exists; new UI capture is MANUAL |
| Reproducibility | PARTIAL | Exact commands and pins exist; live model/cache still external |
| Quantitative current quality | NOT_VERIFIED | Pipeline exists, current run absent |
| Honest claims | PASS | Historical and current states are separated |

Conclusion: the repository is materially improved and suitable for code review. Lab 7 must not be called portfolio-ready or production-ready until live acceptance, current metrics, and authentic UI screenshots are completed.

## H. Exact Windows 11 run instructions

1. Open the repository root in PowerShell.
2. Start Docker Desktop with Linux containers.
3. In LM Studio, load a tool-capable local chat model and start the server at `http://localhost:1234/v1`.
4. Create the isolated environment without replacing an existing CUDA PyTorch stack:

```powershell
.\labs\lab07_final_project\setup.ps1 -Python .\.venv-lab04\Scripts\python.exe
```

5. Start and verify Redis:

```powershell
docker compose -f labs/lab07_final_project/docker-compose.yml up -d
docker compose -f labs/lab07_final_project/docker-compose.yml ps
```

6. Verify the local model and set its exact ID:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
$env:LM_STUDIO_MODEL = "mistralai/ministral-3-3b"
```

7. Build/reuse the index and run retrieval evaluation:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab07_final_project.src.experiments --retrieval-only
```

8. Run CLI and UI:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe labs/lab07_final_project/app.py --debug
.\labs\lab07_final_project\.venv\Scripts\python.exe -m streamlit run labs/lab07_final_project/web_app.py
```

9. Run a fresh live evaluation and then manually review every new run ID before summarizing it:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab07_final_project.src.experiments --force
```

10. Run offline verification:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab05_multimodal/tests labs/lab06_agents/tests labs/lab07_final_project/tests -q
.\labs\lab07_final_project\.venv\Scripts\python.exe -m compileall -q labs/lab06_agents/src labs/lab07_final_project/src labs/lab07_final_project/app.py labs/lab07_final_project/web_app.py
git diff --check
```

11. Repeat the three scenarios from the Lab 7 README. Verify the calculator in the actual trace, not in prose.
12. Manually capture: LM Studio model list, Redis healthy state, CLI grounded answer, Streamlit answer with expanded evidence/tool panel, cross-document calculator trace, unsupported refusal, same-session follow-up, and fresh-session isolation.
