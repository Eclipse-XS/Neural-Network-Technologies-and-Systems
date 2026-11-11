# Scientific Evaluation Datasets: Governance and Provenance

## 1. Governance Architecture

The evaluation suite is partitioned into three explicit governance tiers:

1. **Historical Regression Benchmark Suite**:
   - `frozen_eval_100.json` (100 queries, SHA256: `e729624ca014c04d455518b3e73d2063e01b4782039a0919b173361b2e03b1e4`)
   - `dev_eval_50.json` (50 queries, SHA256: `91da06257a2e41a7fd39cd29860867cc55006007a766881f94721fa0a271f1ff`)
   - `acceptance_eval_50.json` (50 queries, SHA256: `db7f82aa2a4b84e29f916b58ceec2bf5569664932a4f4d32d0f0ff2bf7f3d06a`)
   - *Status*: **Historical Regression Data**. These sets were exposed during iterative optimization and served to identify initial architectural bottlenecks (cross-document routing, token truncation, bilingual stemming). They are retained for regression tracking and baseline comparisons.

2. **Prior Final Holdouts & Diagnostic Remediation Sets**:
   - `final_acceptance_holdout_50.json` (`v1`, 50 queries, SHA256: `663700c2bbf5390cc88800776554b0a8fa1471e733181cb301d8243810a9e1b9`)
   - `final_acceptance_holdout_v2_50.json` (`v2`, 50 queries, SHA256: `b4b7398037d8dd9ee43af8732683331eaaeb5822a1b888b99e01b47be3307696`)
   - `final_acceptance_holdout_v3_50.json` (`v3` duplicate, 50 queries, SHA256: `bd936e38acba5342170510bda0b1fbe075e7708d0279d5c2c3034f6e488ffad1`) — **`HISTORICAL_DUPLICATE_OF_V1_NON_INDEPENDENT`**: retained solely for historical provenance and excluded from independent acceptance evaluation.
   - `final_acceptance_holdout_v3_independent_50.json` (`v3_independent`, 50 queries, SHA256: `b064a2e4f36df05ed5a1137e8e13aca68042f0afc6d1301f80fe3e0ca041f901`) — independent holdout used during PR #4 diagnostic & remediation evaluation (`generation_results_v3.json`).

3. **Frozen Unseen Independent Acceptance Holdout (`v4_independent`)**:
   - `final_acceptance_holdout_v4_independent_50.json` (50 queries, SHA256: `01ae9c5c97308668054d6c2fc5df2b59f2ed5d835626dfc8fef047eb1b30d377`)
   - *Status*: **Frozen Unseen Holdout**. Authored with 50 distinct source-grounded queries (10 `exact_fact`, 10 `paraphrase`, 10 `cross_document`, 10 `numerical_reasoning`, 10 `unsupported`; 25 EN / 25 UK) without parenthetical operand spoilers, audited via `evaluation/audit_dataset_duplicates.py` against all 7 prior sets (**0 duplicates / 0 flagged issues**), and SHA-256 frozen in `dataset_integrity_manifest.json` prior to live evaluation (`generation_results.json` / `generation_results_v4.json`). No code or prompt tuning was performed after evaluating `v4`.

---

## 2. Dataset Schema & Evidence Grounding

Every query across all datasets complies with the verified schema:

```json
{
  "id": "final_001",
  "category": "exact_fact",
  "language": "en",
  "query": "Which vector database and relational database were chosen in ADR 001?",
  "expected_answer": "Redis Stack (Search & JSON) was chosen for vector storage and SQLite with WAL for application state.",
  "relevant_doc_ids": [
    "adr_001_vector_and_storage_architecture.md"
  ],
  "target_sections": [
    "## Decision"
  ],
  "evidence_spans": [
    "**Redis Stack (Search & JSON)** as the vector database:",
    "**SQLite with WAL (Write-Ahead Logging)** for application state and chat memory:"
  ],
  "gold_passage_ids": [
    "chunk-8f19da52e257850a"
  ],
  "expected_tool_calls": [
    "search_knowledge"
  ],
  "refusal_expected": false
}
```

### Invariants:
1. **Verbatim Evidence**: For every non-refusal query, every string in `evidence_spans` MUST exist verbatim in the target document. This invariant is audited automatically by `evaluation/validate_annotations.py`.
2. **Deterministic Passage Identifiers**: `gold_passage_ids` correspond to stable SHA-256 chunk digests computed during document parsing.
3. **Calibrated Refusal**: Queries with `refusal_expected: true` MUST have empty `evidence_spans`, `target_sections`, and `relevant_doc_ids`.

---

## 3. Cryptographic Integrity Manifest

| File | Queries | Supported | Refusal | Verbatim Spans | Gold Passages | SHA256 |
|---|---|---|---|---|---|---|
| `frozen_eval_100.json` | 100 | 85 | 15 | 172 | 205 | `e729624ca014c04d455518b3e73d2063e01b4782039a0919b173361b2e03b1e4` |
| `dev_eval_50.json` | 50 | 43 | 7 | 87 | 67 | `91da06257a2e41a7fd39cd29860867cc55006007a766881f94721fa0a271f1ff` |
| `acceptance_eval_50.json` | 50 | 42 | 8 | 85 | 138 | `db7f82aa2a4b84e29f916b58ceec2bf5569664932a4f4d32d0f0ff2bf7f3d06a` |
| `final_acceptance_holdout_50.json` | 50 | 40 | 10 | 57 | 53 | `663700c2bbf5390cc88800776554b0a8fa1471e733181cb301d8243810a9e1b9` |
| `final_acceptance_holdout_v2_50.json` | 50 | 42 | 8 | 56 | 56 | `b4b7398037d8dd9ee43af8732683331eaaeb5822a1b888b99e01b47be3307696` |
| `final_acceptance_holdout_v3_50.json` *(historical duplicate)* | 50 | 40 | 10 | 57 | 53 | `bd936e38acba5342170510bda0b1fbe075e7708d0279d5c2c3034f6e488ffad1` |
| `final_acceptance_holdout_v3_independent_50.json` *(diagnostic)* | 50 | 40 | 10 | 53 | 52 | `b064a2e4f36df05ed5a1137e8e13aca68042f0afc6d1301f80fe3e0ca041f901` |
| `final_acceptance_holdout_v4_independent_50.json` *(frozen unseen)* | 50 | 40 | 10 | 66 | 58 | `01ae9c5c97308668054d6c2fc5df2b59f2ed5d835626dfc8fef047eb1b30d377` |
