# Scientific Reconciliation of Historical Baseline Evaluation and Discrepancy Analysis

**Document Version:** 2.0.0 (Scientifically Reconciled)  
**Artifact References:**  
- `historical_generation_case_review.json` (Detailed case-by-case human reviewer audit)  
- `repaired_historical_baseline_evaluation.json` (Automated script evaluation)  
- `reconciled_historical_baseline_audit.json` (Full 50-query reconciled audit matrix)  
**Dataset:** `final_acceptance_holdout_50.json` (50 queries: 40 answerable, 10 refusal)  
**Hardware & Runtime:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GiB VRAM), Python 3.13.5, PyTorch 2.14.1+cu130, LM Studio (`mistralai/ministral-3-3b`), Redis Stack 7.4.0  

---

## 1. Executive Reconciliation & Ground Truth

A complete reconciliation across all 50 recorded benchmark turns was performed to eliminate the internal contradiction between the human reviewer audit (which found **42 factually correct/adherent answers**) and the automated evaluation script (which scored **13 correct answers**).

### 1.1 Root Cause of the Discrepancy
1. **Evaluator False Negatives (19 queries):** The automated evaluator rejected mathematically and factually sound answers due to rigid string-matching heuristics, vocabulary ratios, and strict token overlap requirements.
2. **Deceptive Denominator Rejection:** The automated evaluation script originally computed factual correctness over $N = 35$ by dropping the 5 execution failures. An honest scientific denominator requires evaluating over all $N = 40$ answerable queries (or all $N = 50$ total queries). The honest automated baseline is **13 / 40 (32.5%)** and **23 / 50 (46.0%)**, **never 13 / 35 (37.1%)**.
3. **Genuine Model Deficiencies (8 queries):**
   - **5 Execution Failures (Timeouts):** Cross-document queries (`final_021`, `final_022`, `final_025`, `final_027`, `final_030`) failed due to 120s timeouts on sequential tool-calling loops.
   - **3 Semantic Defects:** Hallucinations or retrieval errors (`final_013` CFG inversion, `final_019` runbook path hallucination, `final_020` diffusion formula error).

---

## 2. Reconciled Metrics Table

| Metric | Reviewer Manual Judgment | Automated Evaluator (Honest Denominator) | Automated Evaluator (Deceptive 35 Denominator) | Scientific Ground-Truth Consensus |
| :--- | :---: | :---: | :---: | :---: |
| **Total Evaluation Scope** | 50 queries | 50 queries | 45 executed turns | 50 queries |
| **Execution Failures (Timeouts)** | 5 / 50 (10.0%) | 5 / 50 (10.0%) | *Excluded from stats* | 5 / 50 (10.0%) |
| **Answerable Factual Correctness** | **32 / 40 (80.0%)** | **13 / 40 (32.5%)** | 13 / 35 (37.1%) [INVALID] | 13 automated consensus, 19 evaluator false negatives |
| **Unsupported Refusal Rate** | **10 / 10 (100.0%)** | **10 / 10 (100.0%)** | 10 / 10 (100.0%) | 10 / 10 (100.0%) |
| **Overall Dataset Competence** | **42 / 50 (84.0%)** | **23 / 50 (46.0%)** | 23 / 45 (51.1%) [INVALID] | 23 verified consensus (46.0%) |
| **Calculator Execution Rate** | 7 / 10 (70.0%) called | 7 / 10 (70.0%) called | 7 / 10 (70.0%) called | 7 / 10 called tool, 10/10 math correct in text |
| **Citation Validity Rate** | 36 / 40 (90.0%) | 22 / 40 (55.0%) | 22 / 35 (62.9%) [INVALID] | 22 strict markup consensus |

---

## 3. Confusion Matrix: Reviewer vs. Automated Evaluator

```
                             Automated Evaluator
                         Correct          Incorrect / Failed
Reviewer Correct           23                     19          (Evaluator False Negatives)
Reviewer Incorrect          0                      8          (3 Semantic Defects + 5 Timeouts)
```

- **Consensus Correct (23 / 50 = 46.0%):** Both human reviewer and automated evaluator verified the answer as factual, properly cited, and supported by corpus evidence (13 answerable queries + 10 refusal queries).
- **Evaluator False Negatives (19 / 50 = 38.0%):** Human reviewer verified factual correctness from corpus evidence, but automated evaluator scored 0:
  - *Numerical queries (10 queries, `final_031`–`040`):* Model computed arithmetic correctly in natural language text without calling the external `calculate` function or with format variations.
  - *Ukrainian terminology & morphology (9 queries, e.g. `final_003`, `final_006`, `final_008`, `final_011`, `final_012`, `final_017`, `final_018`, `final_029`):* Automated evaluator rejected Ukrainian answers due to substring mismatches, foreign token penalties, or non-verbatim synonymy.
- **Evaluator False Positives (0 / 50 = 0.0%):** Automated evaluator never credited an answer that the human reviewer determined to be factually incorrect.
- **Consensus Defective / Failed (8 / 50 = 16.0%):**
  - 5 Execution Timeouts (`final_021`, `final_022`, `final_025`, `final_027`, `final_030`): Both agree the model failed to produce a valid response.
  - 3 Factual Hallucinations (`final_013`, `final_019`, `final_020`): Both agree the model generated factually erroneous claims.

---

## 4. Denominator Integrity Rules

1. **Rule 1 (Never omit execution failures):** In any retrieval or generation benchmark, an execution failure (such as an HTTP client timeout or uncaught exception) is a system failure and must be counted as 0 in the numerator while remaining in the denominator.
2. **Rule 2 (Report explicit subsets):** When citing factual correctness, report both:
   - Factual accuracy on answerable questions: $X / 40$.
   - Refusal accuracy on unanswerable questions: $Y / 10$.
   - Combined benchmark score: $(X + Y) / 50$.
