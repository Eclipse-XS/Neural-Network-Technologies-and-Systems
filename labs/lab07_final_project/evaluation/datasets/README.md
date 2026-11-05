# Scientific Evaluation Datasets: Governance and Provenance

## 1. Governance Architecture

The evaluation suite is partitioned into two distinct governance tiers:

1. **Historical Regression Benchmark Suite**:
   - `frozen_eval_100.json` (100 queries, SHA256: `bbf92c004ad91e2247b638e0f1860c2ae47f1aeae41a89495ca3cc7097d36eb3`)
   - `dev_eval_50.json` (50 queries, SHA256: `68ea707746e5b0510bc26af9ab6a8ee3aaacbf513d075e538ab23e806ad0453e`)
   - `acceptance_eval_50.json` (50 queries, SHA256: `308eee463285dd300d2f68079d19c79607c457461134859d81a9e47581bf96c2`)
   - *Status*: **Historical Regression Data**. These sets were exposed during iterative optimization and served to identify initial architectural bottlenecks (cross-document routing, token truncation, bilingual stemming). They are retained for regression tracking and baseline comparisons.

2. **Independent Untouched Final Acceptance Holdout**:
   - `final_acceptance_holdout_50.json` (50 queries, SHA256: `1493c4ea5093bae5cd501c1349beb0513305af98582c7891f1232be7536b82cc`)
   - *Status*: **Untouched Final Holdout**. Authored independently with verified source-grounded evidence spans. No algorithmic tuning or implementation heuristics were adjusted against this set.

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

| File | Queries | Supported | Refusal | Verbatim Spans | SHA256 |
|---|---|---|---|---|---|
| `frozen_eval_100.json` | 100 | 85 | 15 | 172 | `bbf92c004ad91e2247b638e0f1860c2ae47f1aeae41a89495ca3cc7097d36eb3` |
| `dev_eval_50.json` | 50 | 43 | 7 | 88 | `68ea707746e5b0510bc26af9ab6a8ee3aaacbf513d075e538ab23e806ad0453e` |
| `acceptance_eval_50.json` | 50 | 42 | 8 | 84 | `308eee463285dd300d2f68079d19c79607c457461134859d81a9e47581bf96c2` |
| `final_acceptance_holdout_50.json` | 50 | 40 | 10 | 57 | `1493c4ea5093bae5cd501c1349beb0513305af98582c7891f1232be7536b82cc` |
