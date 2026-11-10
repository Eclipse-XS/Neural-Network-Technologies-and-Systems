# Scientific Annotation Procedure and Evaluation Protocol
## Ground Truth Methodology for Local Engineering Knowledge Copilot (Lab 7)

**Document Version:** 2.0.0  
**Status:** Approved Standard  
**Target:** Local Agentic RAG System Evaluation  

---

## 1. Objective and Scientific Rationale

Reliable evaluation of Retrieval-Augmented Generation (RAG) systems requires grounded, reproducible, and verifiable evidence standards. Historical reliance on unconstrained token-overlap heuristics (such as 75% bag-of-words overlap) creates false positive risks where off-topic paragraphs sharing technical jargon receive artificial relevance credit.

This document establishes the scientific annotation protocol, ground truth passage identification rules, and dataset governance model for the **Local Engineering Knowledge Copilot**.

---

## 2. Dataset Split Governance

To maintain scientific integrity and prevent data leakage, the evaluation corpus is strictly partitioned into two independent tiers:

### 2.1 Tier 1: Historical Regression Data (100 Queries)
- `frozen_eval_100.json` (100 queries): The full historical regression benchmark.
- `dev_eval_50.json` (50 queries, odd IDs): Historical development split.
- `acceptance_eval_50.json` (50 queries, even IDs): Historical acceptance split.
- **Role**: Preserved for regression tracking, performance history auditing, and ablation comparisons. Exposed during initial optimization.

### 2.2 Tier 2: Independent Untouched Final Acceptance Holdout (50 Queries)
- `final_acceptance_holdout_50.json` (50 queries, SHA256: `663700c2bbf5390cc88800776554b0a8fa1471e733181cb301d8243810a9e1b9`).
- `final_acceptance_holdout_v2_50.json` (50 queries, SHA256: `b4b7398037d8dd9ee43af8732683331eaaeb5822a1b888b99e01b47be3307696`).
- **Role**: **Frozen and untouched final certification holdouts.** Authored independently with verified source-grounded evidence spans. No implementation or algorithmic tuning was performed against these sets.

### Distribution by Category across Splits:
1. **Exact Fact (`exact_fact`)**: Single-hop factual queries requiring exact architectural constants, configuration keys, or quantitative specifications.
2. **Semantic Paraphrase (`semantic_paraphrase`)**: Queries formulated with colloquial phrasing, synonyms, or cross-lingual terms (Ukrainian technical terminology vs. English source headers).
3. **Cross-Document (`cross_document`)**: Multi-hop comparison queries requiring synthesized evidence from two or more distinct laboratory reports or specifications.
4. **Numerical Reasoning (`numerical_reasoning`)**: Queries requiring calculation (e.g. latency deltas, throughput ratios, budget percentages) over retrieved tabular or numerical constants.
5. **Unsupported / Refusal (`unsupported_refusal`)**: Out-of-scope or out-of-distribution queries for which the corpus contains zero supporting evidence; the agent must explicitly refuse.

---

## 3. Evidence Annotation Standard

### 3.1 Stable Chunk Identifier (`gold_passage_ids`)
Every chunk generated during ingestion is assigned a deterministic, immutable identifier computed by `build_chunks_from_document`:
$$\text{chunk\_id} = \text{chunk-}\langle\text{SHA256}(\text{doc\_sha256}, \text{section}, \text{content}, \text{ordinal})[:16]\rangle$$

Examples:
- `chunk-c6cddafce64bf3f5`
- `chunk-8f19da52e257850a`
- `chunk-34c7ef95a32b694b`

### 3.2 Verbatim Evidence Span (`evidence_spans`)
For every supported query, human annotators mark the exact, minimal contiguous string that proves the claim:
- **Rule of Minimality**: The evidence span must not include extraneous narrative sentences.
- **Verbatim Requirement**: The span must match verbatim against the source markdown document, including mathematical notation and unit names.
- **Automated Verification**: Every span is audited against the source documents by `evaluation/validate_annotations.py`. A single non-matching character fails the build.
- **Multi-Document Requirement**: Cross-document queries must provide distinct evidence spans for *each* required source document.

### 3.3 Target Section Header (`target_sections`)
Annotators record the canonical Markdown heading level 2 (`##`) or level 3 (`###`) where the evidence resides.

---

## 4. Multi-Level Relevance Judgment Algorithm

In `evaluation/metrics.py`, relevance is judged hierarchically:

```mermaid
flowchart TD
    A["Retrieved Chunk"] --> B{"Document Scope Match?"}
    B -- No --> F["Irrelevant (Score = 0)"]
    B -- Yes --> C{"Passage ID Exact Match?"}
    C -- Yes --> R["Relevant (Rank = idx)"]
    C -- No --> D{"Verbatim Evidence Span in Chunk?"}
    D -- Yes --> R
    D -- No --> E{"Target Section Match & Substantive (>40 chars)?"}
    E -- Yes --> R
    E -- No --> G{"Substantive Key Term Match?"}
    G -- Yes --> R
    G -- No --> F
```

1. **Document Scope Gate**: If the chunk belongs to an unrelated document, it is immediately disqualified.
2. **Passage ID Match**: If `chunk.chunk_id` matches `gold_passage_ids`, it is marked relevant with 100% confidence.
3. **Verbatim Span Match**: If `evidence_span in normalize(chunk.text)`, it is marked relevant.
4. **Section Alignment**: If `normalize(chunk.section) in target_sections` and the chunk contains substantive prose (>40 chars), it is marked relevant.
5. **Substantive Token Overlap Fallback**: Only used for legacy test items, requiring $\ge 75\%$ overlap on non-stopword tokens.

---

## 5. Automated Audit Verification

The integrity of all evaluation datasets is verified by executing:
```bash
python labs/lab07_final_project/evaluation/validate_annotations.py
```
This utility audits schema adherence, cross-verifies every evidence span against disk markdown files, and generates `dataset_integrity_manifest.json`.
