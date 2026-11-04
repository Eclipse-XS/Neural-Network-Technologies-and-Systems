# Frozen Evaluation Dataset: Knowledge Copilot (100 Queries)

## Overview
This directory contains the frozen, non-synthetic evaluation benchmark for the Local Engineering Knowledge Copilot (`frozen_eval_100.json`).

## Distribution & Composition
The 100-query benchmark is distributed across five query categories and balanced equally between English and Ukrainian:

| Category | Query Count | Purpose & Verification Standard |
|---|---|---|
| **exact_fact** | 25 | Precision fact extraction from specifications, architectural records, and reports. Grounded in exact terms and numbers. |
| **semantic_paraphrase** | 20 | Natural language queries phrased differently from the source text, testing vector and hybrid semantic matching. |
| **cross_document** | 20 | Multi-document synthesis requiring evidence retrieval from multiple independent technical documents. |
| **numerical_reasoning** | 20 | Complex numerical calculations and ratio evaluations requiring AST calculator invocation. |
| **unsupported** | 15 | Negative controls and adversarial queries about missing facts requiring calibrated refusal without hallucination. |
| **Total** | **100** | **50 Ukrainian / 50 English** |

## Data Schema
Each benchmark entry adheres to the following JSON schema:
```json
{
  "id": "eval_001",
  "category": "exact_fact",
  "language": "uk",
  "query": "Яку розмірність ембеддінгів формує модель all-MiniLM-L6-v2 у Lab 3?",
  "expected_answer": "all-MiniLM-L6-v2 формує 384-вимірні нормалізовані embeddings.",
  "relevant_doc_ids": ["lab03_rag_report.md"],
  "gold_chunks": ["all-MiniLM-L6-v2", "384-вимірні embeddings"],
  "expected_tool_calls": [],
  "refusal_expected": false
}
```

## Ground Truth Principles
1. **Zero Hallucination Tolerance**: Ground truth was established strictly from the committed knowledge base corpus (`assets/knowledge/` and `assets/knowledge/engineering/`).
2. **Deterministic Evaluation**: No stochastic prompt-based evaluators are used for gold retrieval matching.
3. **Immutability**: This benchmark is frozen; developers must not modify query targets or relevance labels to inflate metrics.
