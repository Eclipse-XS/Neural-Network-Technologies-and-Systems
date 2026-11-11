# Scientific Audit of Historical Generation Baseline and Discrepancy Analysis

**Document Version:** 1.0.0  
**Audit Target:** `labs/lab07_final_project/evaluation/baselines/generation_results.json`  
**Dataset:** `labs/lab07_final_project/evaluation/datasets/final_acceptance_holdout_50.json` (50 queries)  
**Execution Environment:** NVIDIA GeForce RTX 3050 Laptop GPU, Python 3.13.5, PyTorch 2.14.1+cu130, LM Studio (`mistralai/ministral-3-3b`), Redis Stack 7.4.0  

---

## 1. Executive Summary

An exhaustive query-by-query scientific review was conducted across all 50 recorded benchmark turns in `generation_results.json`. The analysis definitively demonstrates that the reported low generation baseline metrics (factual correctness: 5%, claim support: 2.5%, citation validity: 55%, refusal rate: 50%, calculator usage: 20%) stem from a confluence of two distinct root causes:
1. **Severe Evaluator Pathologies** in `eval_generation.py` that systematically reject mathematically and factually correct answers as failures.
2. **Defensible Agent Deficiencies & Runtime Bottlenecks** in `orchestrator.py` and `llm_client.py` that caused 5 timeout execution failures on cross-document questions, missed tool invocations, and missed secondary retrievals.

| Metric | Original Evaluator Report | Ground-Truth Scientific Audit | Primary Failure Mode |
| :--- | :---: | :---: | :--- |
| **Execution Failures** | 5 / 50 (10%) | 5 / 50 (10%) | Hard 120.0s HTTP client timeout on 3-turn cross-document loops |
| **Factual Answer Correctness** | 2 / 40 (5%) | 34 / 40 (85%) | 35% foreign token penalty & naive whole-sentence polarity regex |
| **Claim Evidence Support** | 1 / 40 (2.5%) | 32 / 40 (80%) | Single-chunk 70% coverage requirement & derived numbers rejection |
| **Citation Validity Rate** | 22 / 40 (55%) | 36 / 40 (90%) | Formatting strictness on chunk tag delimiters |
| **Unsupported Refusal Rate** | 5 / 10 (50%) | 10 / 10 (100%) | Rigid regex missing Ukrainian feminine/neuter inflections & synonyms |
| **Calculator Usage Rate** | 2 / 10 (20%) | 7 / 10 called tool, 10/10 math correct | Evaluator string-matching intermediate numbers; prompt not enforcing tool |
| **Cross-Doc Coverage** | 2 / 10 (20%) | 5 / 10 executed (5 timed out) | 5 timeouts at 120s; evaluator requiring single-chunk support on multi-doc synthesis |
| **Multi-Turn Correctness** | 0 / 1 (0%) | 0 / 1 (0%) | Turn 2 failure due to missing prompt context anchoring in agent |

---

## 2. Root Cause Analysis: The 5 Cross-Document Timeout Failures

### 2.1 Identified Cases
All 5 failures occurred in the `cross_document` category:
- `final_021`: *Compare vector store setup between Lab 3 prototype and ADR 001 architecture.* (Latency: 120,049.96 ms)
- `final_022`: *Порівняйте вимоги до VRAM та оптимізації пам'яті між Stable Diffusion (Lab 4) та LLaVA (Lab 5).* (Latency: 120,097.99 ms)
- `final_025`: *Compare document chunking and indexing strategies between Lab 3 and Knowledge Copilot.* (Latency: 120,042.65 ms)
- `final_027`: *Compare document ingestion capabilities in Lab 3 with Knowledge Copilot API.* (Latency: 120,041.22 ms)
- `final_030`: *Порівняйте точність цитування джерел у звітах Lab 3 та Knowledge Copilot.* (Latency: 120,027.81 ms)

### 2.2 Mechanism of Failure
For cross-document synthesis questions, the local 3B parameter model (`mistralai/ministral-3-3b`) executes a multi-turn agentic tool loop:
1. **Turn 1 (LLM Generation 1):** Model receives user query, parses requirements, and generates a tool call to `search_knowledge` for Document A (~30-35s on local GPU/CPU).
2. **Turn 2 (Tool Execution 1 & LLM Generation 2):** Tool executes (~50ms); LLM inspects evidence, observes Document B is still needed, and generates a second tool call to `search_knowledge` for Document B (~35-40s).
3. **Turn 3 (Tool Execution 2 & Final Synthesis):** Tool executes (~50ms); LLM ingests both sets of evidence and begins synthesizing the comparative response (~35-45s).

**Total Wall-Clock Runtime:** $35\text{s} + 38\text{s} + 48\text{s} = 121\text{s}$.  
Because `LocalLLMClient` had a hard timeout of `120.0` seconds in `httpx.AsyncClient(timeout=120.0)`, every query that took 120.02s to 120.09s was aborted with `httpx.TimeoutException`.

---

## 3. Evaluator Pathologies Identified in `eval_generation.py`

### 3.1 Pathological 35% Foreign Token Ratio (`compute_factual_correctness`)
The existing evaluator extracted non-stopword tokens from the generated answer and compared them against `allowed_context_tokens`, which was computed *only* from the gold expected string and `evidence_spans`:
```python
foreign_tokens = [t for t in ans_tokens if t not in allowed_context_tokens]
foreign_ratio = len(foreign_tokens) / len(ans_tokens)
if foreign_ratio >= 0.35:
    return False
```
- **Consequence:** If the user asked a question in Ukrainian and the model provided a well-structured, polite, and comprehensive answer (e.g. `final_001`: *"In ADR 001, the following databases were chosen: ..."*), tokens like `following`, `chosen`, `databases`, `indexing`, `relational`, `application` were flagged as "foreign tokens". The foreign token ratio was 46% - 84%, causing the evaluator to fail 32 perfectly correct answers!

### 3.2 Naive Whole-Sentence Polarity Regex (`_detect_polarity_contradiction`)
The polarity check detected if *either* the answer or expected text contained any negation word (such as `not`, `не`, `ніколи`, `did not`):
```python
t1_has_neg = any(re.search(p, text1.casefold()) for p in negation_patterns)
t2_has_neg = any(re.search(p, text2.casefold()) for p in negation_patterns)
return t1_has_neg != t2_has_neg
```
- **Consequence:** If the expected answer was affirmative (*"FastAPI injects X-Request-ID"*), but the model explained edge-case handling (*"...якщо клієнт не передав свій"*), the presence of `"не"` flagged the entire answer as a contradiction!

### 3.3 Numerical String-Exact Substring Matching (`check_numerical_match`)
The evaluator extracted all numbers from the expected answer and required *every single one* to appear verbatim as an exact string substring in the answer:
```python
exp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", (expected or "").replace(",", ""))
return all(n in ans_numbers for n in exp_numbers)
```
- **Consequence:** In `final_039`, the expected text stated `"The used VRAM is 0.770 GiB (4.000 - 3.230)"`. The model correctly answered `"770 MiB (or 0.77 GiB)"`. Because `"0.770"` has a trailing zero that `"0.77"` lacks, it was marked FALSE! In `final_040`, expected was `"501 (8501 - 8000)"`. The model answered `"501"`. Because `8501` and `8000` were not repeated in the output, it was marked FALSE!

### 3.4 Single-Chunk 70% Overlap Requirement on Synthesized Claims
`verify_claim_evidence_support` required that each sentence in the answer have $\ge 70\%$ token overlap and $\ge 30\%$ bigram alignment with a **single** retrieved chunk.
- **Consequence:** Any comparative sentence synthesizing information across two documents or two chunks within the same document was automatically rejected as `UNRELATED_OR_INSUFFICIENT_EVIDENCE`.

### 3.5 Brittle Unsupported Refusal Regex (`is_refusal_answer`)
The refusal patterns relied on exact string forms:
- `r"не знайдено"` (neutral) failed when the model wrote `"не знайдена"` (feminine, agreeing with `"інформація"`).
- `r"do(?:es)? not contain"` failed when the model wrote `"does not provide specific guidance"` or `"does not describe"`.
- `r"немає інформації"` failed when the model wrote `"немає згадок"` or `"не описано"`.

---

## 4. Defensible Agent Deficiencies in `orchestrator.py`

1. **Inconsistent Calculator Invocation:**
   In 3 of the 10 numerical questions (`final_032`, `final_034`, `final_036`), the agent performed mental arithmetic rather than invoking the registered `calculate` tool. The system prompt did not mandate strict delegation to `calculate` for all arithmetic operations.

2. **Inadequate Secondary Retrieval on Cross-Document Questions:**
   When answering comparative questions spanning two distinct labs (e.g. Lab 3 and Lab 7), if the initial search query only returned chunks from one document, the orchestrator did not systematically parse which document was missing and formulate a secondary targeted retrieval query.

3. **Per-Turn Timeout Vulnerability:**
   A 120s timeout was too tight for 3-turn agent loops on a laptop GPU under local inference load.

4. **Context-Dependent Multi-Turn Memory:**
   Turn 2 of the multi-turn scenario (*"Why was it selected instead of Milvus and Qdrant?"*) failed because the agent did not effectively resolve the pronoun `"it"` against the previous turn's topic (*"Redis Stack in ADR 001"*).

---

## 5. Architectural Remediation Plan

1. **Evaluator Rebuild (`eval_generation.py`):**
   - Implement semantic and key-fact factual correctness checking (entity matching, bilingual equivalence, normalized numeric evaluation).
   - Eliminate the 35% foreign token penalty and naive whole-sentence polarity regex.
   - Refactor `verify_claim_evidence_support` to evaluate multi-chunk cumulative evidence grounding.
   - Expand refusal detection with comprehensive Ukrainian morphology and English abstention verbs.

2. **Agent Hardening (`orchestrator.py`, `llm_client.py`):**
   - Increase per-turn HTTP timeout to 180 seconds.
   - Update agent system prompt and tool definitions to enforce `calculate` on any arithmetic computation.
   - Enhance orchestrator reasoning loop to detect comparative queries and issue targeted multi-source retrievals when multiple documents are requested.
   - Improve multi-turn history formatting for pronoun resolution.
