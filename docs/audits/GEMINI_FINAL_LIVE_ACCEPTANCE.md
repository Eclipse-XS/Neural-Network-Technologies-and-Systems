# Gemini Final Live Acceptance & Quality Assurance Audit (Revised Consistent Version)

Date: 2026-10-30  
Repository: `C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems`  
Branch: `codex/labs-3-4-portfolio-upgrade`  
Initial HEAD: `01ec07f9ce7394365b22e0308fe38e01c30790a8`  
Final HEAD: `01ec07f9ce7394365b22e0308fe38e01c30790a8` (No unauthorized commit, push, or merge created)  
Primary Target: `labs/lab07_final_project/`  
Secondary Targets: `labs/lab06_agents/`, `labs/lab05_multimodal/`  

---

## 1. Executive Summary & Verification Status

**Final Status**: `READY_WITH_DOCUMENTED_LIMITATIONS`

All planned engineering objectives, infrastructure validations, automated testing, Streamlit UI testing, independent retrieval evaluation on held-out data, and academic traceability compliance checks have been executed and verified against real, live backends without synthetic data, mocks, or unauthorized code commits.

| Component | Test / Inspection | Actual Verified Result | Evidence | Status |
|---|---|---|---|---|
| **Local Environment** | PyTorch + CUDA, Redis 6380, LM Studio 1234 | `torch 2.14.1+cu130`, GPU: RTX 3050 Laptop (4 GB), Redis 8.10.3 + Search 8.1.1, LM Studio `mistralai/ministral-3-3b` | `outputs/metadata/environment.json` | **PASS** |
| **Dependency Health** | `pip check` and package imports | 0 broken requirements. Missing `cryptography` & `pyjwt` added to `requirements.txt` & `requirements-ci.txt` | `requirements.txt`, `pip check` | **PASS** |
| **Corpus & Chunker** | SHA-256 integrity, token budget | 4 reports, SHA-256 verified, 112 chunks (73 canonical, 39 quoted model output), max 240 tokens <= 240 budget, 0 truncation violations | `outputs/metadata/embedding.json`, `index.json` | **PASS** |
| **Corpus Contamination** | Searchable vs excluded chunk test | 73 canonical chunks searchable; 39 quoted model transcripts 100% excluded from search candidates | Tested in `task-362` | **PASS** |
| **Vector Store** | Redis FLAT / COSINE schema | 112 indexed vectors (dim=384, float32, cosine), corpus fingerprint verified, idempotent rebuild | `outputs/metadata/index.json` | **PASS** |
| **Held-Out Retrieval** | 30 independent held-out queries (hybrid vs dense vs lexical) | Hybrid: Recall@1 = 0.4286, Recall@3 = 0.6964, Recall@5 = 0.7321, Hit@5 = 0.7857, MRR = 0.6024, Latency = 39.05 ms | `artifacts/portfolio/held_out_eval/` | **PASS** |
| **Agent Execution** | Final live batch (`experiment_ba59e7949f50`) | 16 runs executed on final runtime; 100% runner completion; 0 crashes; 11/12 assistant turns CORRECT, 1 PARTIAL | `outputs/runs/batch.json`, `outputs/runs/runs.jsonl` | **PASS** |
| **Grounded Citations** | Exact chunk ID membership and syntactic unwrap | 0 invalid citations across compliant answers; abstains on unsupported questions without hallucinating citations | `agent.py`, `validate_citations` unit tests | **PASS** |
| **Arithmetic Policy** | Multi-step calculation | Cross-document operands verified from evidence | `outputs/traces/` | **PASS** |
| **Streamlit Web UI** | AppTest E2E live session & chat | Initial render, document catalog, live LLM question-answering with markdown response & tool trace, session reset | `tests/test_web_app.py` (3 passed in 76s) | **PASS** |
| **Automated Tests** | Labs 5–7 pytest suite | 74 passed, 1 skipped (0 failed) across 3 lab directories | `pytest` full suite | **PASS** |
| **Code Quality** | Syntax & formatting | `compileall -q`: 0 errors; `pip check`: 0 errors; `git diff --check`: 0 whitespace issues | CLI execution outputs | **PASS** |
| **CI Status** | Workflow validation | Offline CI workflow valid and locally verified (`LOCALLY_VALIDATED`); pending remote push | `.github/workflows/labs-6-7-ci.yml` | **LOCALLY_VALIDATED** |
| **Academic Matrix** | Official requirements coverage | Labs 5, 6, and 7 requirements satisfied with real code and verified evidence | Section 8 below | **PASS** |

---

## 2. Infrastructure & Environment Details

| Field | Measured Value |
|---|---|
| OS | Windows 11 Home / Pro (10.0.26300-SP0) |
| Python Executable | `C:\Users\Eclipse\PycharmProjects\neural-network-technologies-and-systems\labs\lab07_final_project\.venv\Scripts\python.exe` (Python 3.13.5) |
| PyTorch / CUDA | `torch 2.14.1+cu130` with CUDA 13.0 |
| GPU Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4.000 GiB VRAM) |
| Redis Server | Version `8.10.3` running inside WSL2 on port `6380` |
| Redis Modules | `search` (8.1.1), `vectorset` (1.0.0) |
| LM Studio Endpoint | `http://localhost:1234/v1` (OpenAI Chat Completions compatible) |
| Active Local LLM | `mistralai/ministral-3-3b` |
| Embedding Model | `intfloat/multilingual-e5-small` |
| Embedding Revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Embedding Dimension | 384 (normalized, Cosine distance) |
| Docker Status | Broken Windows socket (`sailor-ingest.sock`); safely bypassed via native WSL2 Redis |
| External Port 8000 | User's unrelated Uvicorn application strictly untouched |

---

## 3. RAG Ingestion & Token-Budget Audit

- **Knowledge Documents**: 4 markdown reports (`lab03_rag_report.md`, `lab04_stable_diffusion_report.md`, `lab05_multimodal_report.md`, `lab06_agents_report.md`).
- **Corpus Integrity**: All 4 files match the SHA-256 hashes recorded in `manifest.json`.
- **Chunk Distribution**:
  - `lab03`: 4 chunks (canonical)
  - `lab04`: 26 chunks (canonical)
  - `lab05`: 33 chunks (canonical)
  - `lab06`: 49 chunks (10 canonical report chunks, 39 quoted model transcripts tagged `quoted_model_output`)
  - **Total indexed in Redis**: 112 chunks.
- **Corpus Contamination Breakdown**:
  - Searchable chunks (`evidence_quality == 'canonical_report'`): **73 chunks**.
  - Excluded model transcripts (`evidence_quality == 'quoted_model_output'`): **39 chunks**.
  - Standard retrieval query tests confirmed that **0%** of quoted model output chunks are returned in search results.
- **Token Budgeting**:
  - Model sequence limit: 512 tokens.
  - Configured chunk budget: 240 tokens (with safety margin for `passage: ` prefix).
  - Maximum observed chunk token count: 240 tokens.
  - **Truncation violations: 0 (0.0%)**.

---

## 4. Retrieval Evaluation: Independent Held-Out Benchmark

### 4.1. Methodological Correction (Recall@k Monotonicity)
Earlier evaluation accidentally calculated Recall@3 as binary hit rate (`any(rank <= 3)` = 1.0) while Recall@5 was calculated as fractional evidence recall ($\frac{\sum \mathbb{I}(\text{rank} \le 5)}{|E|}$), causing a non-monotonic anomaly (0.7500 > 0.7321).
In the corrected evaluation:
$$\text{Recall}@k = \frac{1}{|Q|}\sum_{q \in Q} \frac{\sum_{e \in E_q} \mathbb{I}(\text{rank}(e) \le k)}{|E_q|}$$
is evaluated consistently across $k \in \{1, 3, 5\}$. Binary Hit@k is reported separately as $\text{Hit}@k = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{I}(\min_{e \in E_q} \text{rank}(e) \le k)$.
A regression unit test (`test_retrieval_monotonicity_invariant`) was added to `test_evaluation.py` to prevent future metric divergence.

### 4.2. Verified Results on Independent Held-Out Set (30 Queries, N=28 Evaluated, 2 Unsupported)

| Configuration | Evidence Recall@1 | Evidence Recall@3 | Evidence Recall@5 | Evidence Hit@5 | Evidence MRR | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Lexical (BM25)** | 0.5179 | 0.6250 | 0.7143 | 0.7500 | 0.6042 | **0.87 ms** |
| **Dense (Multilingual E5)** | 0.3929 | 0.5893 | 0.7321 | 0.7857 | 0.5536 | 38.15 ms |
| **Hybrid (E5 + BM25 RRF)** | **0.4286** | **0.6964** | **0.7321** | **0.7857** | **0.6024** | 39.05 ms |

*Historical baseline (Dense MiniLM):* Evidence Recall@5 = 0.1250, MRR = 0.3690.

**Analysis & Trade-offs:**
- **Monotonicity**: Evidence Recall strictly increases: $0.4286 \le 0.6964 \le 0.7321$.
- **Hybrid Advantage**: Hybrid RRF achieves the highest Recall@3 (0.6964 vs 0.5893 for Dense) and matches peak Recall@5 (0.7321), effectively fusing exact token matches (BM25) with semantic embeddings (E5).
- **Latency**: After standard warmup, Dense takes 38.15 ms and Hybrid takes 39.05 ms (Dense + BM25 + rank merge), eliminating the temporary CPU thermal anomaly observed earlier.

---

## 5. Agent Evaluation: Final Live Regression Batch

Evaluated on the final runtime codebase with updated citation unwrapping and operand extraction.

### 5.1. Rigorous Accounting of Runs and Turns
- Total Scenario Definitions: **11**
- Multi-turn Scenarios: `memory` contains 2 sequential user turns (`memory_0` and `memory_1`).
- Total Assistant Turns Evaluated: **12 turns**
- Baseline Plain LLM Control Turns: **4 turns** (`lab4`, `cross`, `multi`, `unsupported`)
- **Total Runs in Batch**: **16 runs**

### 5.2. Scenario-by-Scenario Evaluation Table

| Turn # | Scenario ID | Variant | Turn Type | Tools Used | Citations | Status | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `role` | assistant | no-tool | None | None | **CORRECT** | Accurately describes capabilities over indexed reports |
| 2 | `lab4` | assistant | RAG | `search_knowledge` | `[lab04#chunk-021f33e4ebbbb4ea]` | **CORRECT** | Exact model `runwayml/stable-diffusion-v1-5`, rev `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| 3 | `lab4` | plain | baseline | None | None | **INCORRECT** | Fabricated nonexistent report ID `lab_4_demo` |
| 4 | `lab3` | assistant | semantic | `search_knowledge` | `[lab03#chunk-24bb056971f36561]` | **CORRECT** | Identifies Recall@5 multi-record retrieval limitation |
| 5 | `lab5` | assistant | RAG | `search_knowledge` | `[lab05#chunk-e14761fa8573f0b1]` | **CORRECT** | Identifies hallucinated bounding box `[0.542, 0.544, 0.856, 0.731]` |
| 6 | `cross` | assistant | cross-source | `search_knowledge` (x2) | `[lab04#chunk-021f33e4ebbbb4ea]`, `[lab05#chunk-70830b7bfc138ce1]`, `[lab05#chunk-9e6f1c9e586d6c4f]`, `[lab05#chunk-f5f090b490023afe]` | **CORRECT** | Accurately contrasts SD text-to-image pipeline with LLaVA VQA |
| 7 | `cross` | plain | baseline | None | None | **INCORRECT** | Generic transformer claims without report facts |
| 8 | `multi` | assistant | multi-step | `list_knowledge_sources`, `search_knowledge` (x4) | `[lab04#chunk-ab2b4d9921a7aa9c]`, `[lab05#chunk-b871e9d71bd9227b]` | **PARTIAL** | Correctly found Lab 4 = 13; for Lab 5 cited 9 temperature experiments instead of total 17 unique inferences |
| 9 | `multi` | plain | baseline | None | None | **INCORRECT** | Fabricated 5 generations and non-existent CSV |
| 10 | `memory_0` | assistant | memory (t1) | `list_knowledge_sources`, `search_knowledge` | `[lab06#chunk-7fb62281ab572271]` | **CORRECT** | Exact model `mistralai/ministral-3-3b` identified |
| 11 | `memory_1` | assistant | memory (t2) | `search_knowledge` | `[lab06#chunk-5927646dd5965acf]` | **CORRECT** | Resolves `it` to Mistral 3B; identifies tool selection weakness |
| 12 | `isolation` | assistant | isolation | None | None | **CORRECT** | Ambiguity guard halts pronoun without antecedent in fresh session |
| 13 | `unsupported` | assistant | unsupported | `search_knowledge` | `[lab04#chunk-6d42e4df2e3f185b]`, `[lab04#chunk-528416671a85a1c7]` | **CORRECT** | Grounded refusal: explicitly states outdoor temperature is absent |
| 14 | `unsupported` | plain | baseline | None | None | **CORRECT** | Plain LLM also acknowledged lack of weather data |
| 15 | `prompt_injection` | assistant | security | None | None | **CORRECT** | Strict safety refusal; 0 unauthorized tools called |
| 16 | `sources` | assistant | discovery | `list_knowledge_sources` | None | **CORRECT** | Accurately lists Labs 3, 4, 5, 6 |

### 5.3. Verified Accounting Summary
- **Assistant Turns (N = 12)**:
  - **CORRECT**: 11 / 12 (**91.67%**)
  - **PARTIAL**: 1 / 12 (**8.33%**) (`multi`)
  - **INCORRECT**: 0 / 12 (**0.00%**)
  - **Invalid Citation Rate**: 0.0% (all cited chunks match current-run evidence).
- **Baseline Plain LLM Turns (N = 4)**:
  - **CORRECT**: 1 / 4 (25.00%) (`unsupported` refusal)
  - **INCORRECT / FABRICATED**: 3 / 4 (75.00%) (`lab4`, `cross`, `multi`)

---

## 6. Streamlit Functional E2E Verification

Verified with `streamlit.testing.v1.AppTest` connected to live services:
- **Test 1 (`test_streamlit_initial_render_and_catalog`)**: Initial page load, session ID display, 4 indexed source titles in sidebar. **PASS**.
- **Test 2 (`test_streamlit_live_e2e_question_and_answer`)**: Live query `"What model was used in Lab 6?"` executed through Agent, Redis search, and local Mistral 3B. Assistant markdown response rendered with `mistralai/ministral-3-3b` and expandable tool trace json containing `search_knowledge`. **PASS**.
- **Test 3 (`test_streamlit_session_controls`)**: `New session` button resets session state and generates a fresh UUID. **PASS**.
- **Execution Time**: 3 passed in 76.13s (includes initial cold-start model weight loading).

---

## 7. Automated Test Suite Summary

Executed command:
```powershell
python -m pytest labs/lab05_multimodal/tests labs/lab06_agents/tests labs/lab07_final_project/tests -q
```
- `labs/lab05_multimodal/tests`: 7 passed
- `labs/lab06_agents/tests`: 32 passed, 1 skipped (manual interactive test)
- `labs/lab07_final_project/tests`: 35 passed (32 unit/regression + 3 Streamlit live E2E)
- **Total**: **74 passed, 1 skipped, 0 failed in 78.66s**.

Integrity checks:
- `compileall -q`: **PASS** (0 syntax errors).
- `pip check`: **PASS** (`No broken requirements found`).
- `git diff --check`: **PASS** (0 whitespace/formatting issues).
- **CI Status**: `LOCALLY_VALIDATED` (offline workflows execute cleanly in local environment; pending Git push / PR).

---

## 8. Academic Compliance & Traceability

### Lab 5 — LLaVA Multimodal VQA (Variant 1)
- Model `llava-hf/llava-interleave-qwen-0.5b-hf`, 19 experimental rows, temperature/token comparisons, hallucination analysis: **PASS**.
- Limitation: Screenshots of GUI execution remain **MANUAL** documentation assets.

### Lab 6 — OpenAI Agents SDK (Variant 3)
- OpenAI Agents SDK with local LM Studio `mistralai/ministral-3-3b`, SQLite session memory, functional workspace tools, autonomous vs baseline comparison: **PASS**.

### Lab 7 — RAG + Agent Integration (Variant 1)
- CLI and Streamlit interfaces, Redis vector index, multilingual E5 chunking (0 truncations), hybrid BM25 retrieval, grounded answers, tool calling, isolated memory, plain LLM comparison: **PASS**.

---

## 9. Documented Limitations

1. **Deterministic Arithmetic Policy Recovery**:
   When small 3B models omit the `calculate` tool on multi-operand arithmetic, deterministic runtime policy verifies operands from evidence and executes `safe_calculate`, logging `policy_enforced=True`.
2. **Multi-evidence Count Sensitivity**:
   In complex reports containing both sub-experiments (e.g. 9 temperature tests) and total unique inference counts (17), 3B models may occasionally retrieve the sub-experiment count instead of the grand total if the prompt does not specify "grand total".
3. **Hardware & Compute**:
   RTX 3050 Laptop GPU (4 GB VRAM) requires running LM Studio on GPU while sentence transformers embeddings run on CPU to avoid VRAM OOM.
4. **Manual Screenshot Capture**:
   Streamlit and CLI live functional testing is automated via `AppTest`; physical browser screenshots are manual verification tasks.

---

## 10. Conclusion

The system is verified, mathematically consistent, reproducible, and ready for portfolio presentation. All metrics and claims are grounded in raw artifacts without fabrication.
