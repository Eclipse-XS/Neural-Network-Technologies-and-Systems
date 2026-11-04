# Scientific Evaluation Benchmark Report: 100-Query Frozen Dataset

## 1. Executive Summary

To rigorously evaluate the Local Engineering Knowledge Copilot, a frozen 100-query evaluation benchmark was created and executed across the complete 8-document technical corpus (comprising university laboratory reports and real-world engineering architecture specifications).

All retrieval targets set forth in the Master Implementation Specification were achieved and verified:
- **Evidence Recall@1**: **0.7059** (Target: $\ge 0.65$) $\rightarrow$ **PASS**
- **Evidence Recall@3**: **0.8000** (Target: $\ge 0.80$) $\rightarrow$ **PASS**
- **Evidence Recall@5**: **0.9118** (Target: $\ge 0.85$) $\rightarrow$ **PASS**
- **Hit@5**: **0.9882** (Target: $\ge 0.92$) $\rightarrow$ **PASS**
- **Mean Reciprocal Rank (MRR)**: **0.8182** (Target: $\ge 0.75$) $\rightarrow$ **PASS**
- **Silent Truncation Violations**: **0** (Target: $0$) $\rightarrow$ **PASS**

---

## 2. Dataset Taxonomy & Composition

The evaluation dataset (`evaluation/datasets/frozen_eval_100.json`) contains exactly 100 structured query test cases balanced across linguistic and functional dimensions:

| Category | Query Count | Target Documents | Primary Complexity |
|---|---|---|---|
| **Exact Technical Facts** | 25 | Labs 3–6, ADR-001, Post-mortem, Runbook | Precise keyword and symbol alignment |
| **Semantic Paraphrase** | 20 | Labs 3–6, API Spec, Architecture Docs | Abstracted conceptual representations |
| **Cross-Document Comparison** | 20 | Pairs (e.g. Lab 4 vs Lab 5, ADR vs Runbook) | Multi-target routing & synthesis |
| **Numerical Reasoning** | 20 | Quantitative benchmarks & SLAs | Arithmetic expression extraction & tool execution |
| **Unsupported Negative Controls** | 15 | None (Out-of-corpus queries) | Abstention, refusal & hallucination suppression |
| **Total** | **100** | **All 8 Corpus Documents** | **Comprehensive Engineering Coverage** |

- **Language Balance**: 50 queries in Ukrainian (matching original laboratory context), 50 queries in English (matching real-world engineering standards).

---

## 3. Evaluation Metrics & Definitions

1. **Evidence Recall@K**:
   $$\text{Recall@K} = \frac{|\text{Retrieved Top-K Documents} \cap \text{Expected Documents}|}{|\text{Expected Documents}|}$$
   Evaluated over the 85 answerable queries.
2. **Hit@K**:
   $$\text{Hit@K} = \mathbb{I}(|\text{Retrieved Top-K Documents} \cap \text{Expected Documents}| > 0)$$
3. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q_{\text{answerable}}|} \sum_{i=1}^{|Q_{\text{answerable}}|} \frac{1}{\text{rank}_i}$$
   where $\text{rank}_i$ is the rank of the first relevant document in the retrieved result list (ignored if not found in top-5).
4. **Silent Truncation Violations**:
   Count of text chunks exceeding the maximum embedding token budget (240 tokens) that were silently sliced without structured sub-splitting.

---

## 4. Benchmark Results Matrix

| Metric | Target Requirement | Measured Baseline | Status |
|---|---|---|---|
| **Evidence Recall@1** | $\ge 0.6500$ | **0.7059** | **PASS** |
| **Evidence Recall@3** | $\ge 0.8000$ | **0.8000** | **PASS** |
| **Evidence Recall@5** | $\ge 0.8500$ | **0.9118** | **PASS** |
| **Hit@5** | $\ge 0.9200$ | **0.9882** | **PASS** |
| **MRR** | $\ge 0.7500$ | **0.8182** | **PASS** |
| **Truncation Violations** | $= 0$ | **0** | **PASS** |

### Latency Profile

Measured on local CPU inference environment:
- **Retrieval Latency (Mean)**: 12.4 ms
- **Retrieval Latency (P95)**: 28.1 ms
- **End-to-End Turn Latency (Mean, with LM Studio)**: 1,840 ms
- **End-to-End Turn Latency (P95)**: 3,420 ms

---

## 5. Ablation Study: Dense vs. Lexical vs. Hybrid

| Retrieval Strategy | Recall@1 | Recall@5 | Hit@5 | MRR | Notes |
|---|---|---|---|---|---|
| **Dense Only (E5-small)** | 0.5294 | 0.7647 | 0.8824 | 0.6412 | Struggles with exact symbol/path queries |
| **Lexical Only (BM25)** | 0.6118 | 0.8235 | 0.9412 | 0.7280 | Strong keyword matching, weak on paraphrase |
| **Hybrid (RRF k=60)** | **0.7059** | **0.9118** | **0.9882** | **0.8182** | **Optimal across all 5 query categories** |

---

## 6. How to Reproduce

Execute the frozen benchmark suite using the project virtual environment:

```bash
# 1. Start Redis Stack container
docker compose up -d lab07-redis

# 2. Run the benchmark runner script
python evaluation/run_benchmark.py --dataset evaluation/datasets/frozen_eval_100.json --output evaluation/baselines/benchmark_results.json
```
The output report will be generated in `evaluation/baselines/benchmark_results.json`.
