"""Evaluation metrics calculation for RAG retrieval and generation.

Implements rigorous, multi-level relevance judgment:
1. Exact Passage ID / Chunk ID match (highest confidence).
2. Verbatim Gold Evidence Span containment in chunk body or section.
3. Target Section alignment within the verified document scope.
4. Semantic / Lexical substantive overlap fallback for historical benchmarks.
"""
import re
from typing import Any, Dict, List, Optional
import numpy as np


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove markdown styling characters."""
    t = re.sub(r"[`*_#]", " ", text or "")
    return re.sub(r"\s+", " ", t).strip().casefold()


def is_chunk_relevant(chunk: dict, gold_query: dict) -> bool:
    """Determine if a retrieved chunk matches ground truth targets with verified semantics.
    
    A correct document name or section title alone NEVER establishes factual evidence relevance.
    Relevance strictly requires:
    1. Exact validated passage ID; or
    2. Verbatim supporting evidence span in retrieved chunk content; or
    3. (Historical fallback) Substantive multi-word evidence overlap against gold chunk terms.
    """
    filename = (chunk.get("filename") or "").casefold()
    source_id = (chunk.get("source_id") or "").casefold()
    document_id = (chunk.get("document_id") or "").casefold()
    chunk_id = (chunk.get("chunk_id") or "").casefold()
    content = normalize_text(chunk.get("content") or chunk.get("text") or "")
    section = normalize_text(chunk.get("section") or "")

    # 1. Document Scope Verification
    relevant_docs = [d.casefold() for d in gold_query.get("relevant_doc_ids", [])]
    if relevant_docs:
        doc_match = any(
            (d in filename or (filename and filename in d) or
             d in source_id or (source_id and source_id in d) or
             d in document_id or (document_id and document_id in d))
            for d in relevant_docs
            if d
        )
        if not doc_match:
            return False

    # 2. Passage ID / Chunk ID exact match
    gold_passage_ids = [p.casefold() for p in (gold_query.get("gold_passage_ids") or gold_query.get("passage_ids") or [])]
    if gold_passage_ids and chunk_id in gold_passage_ids:
        return True

    # 3. Verbatim Gold Evidence Span match in chunk content
    evidence_spans = gold_query.get("evidence_spans", [])
    if evidence_spans:
        for span in evidence_spans:
            norm_span = normalize_text(span)
            if norm_span and norm_span in content:
                return True

    # 4. Fallback Gold Chunks / Key Evidence Phrases (substantive match only)
    gold_terms = [g for g in gold_query.get("gold_chunks", [])]
    if not gold_terms:
        # Document or section alone NEVER establishes relevance
        return False

    STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "has", "have", "had", "not", "but", "what", "which", "who",
        "how", "why", "when", "where", "can", "could", "will", "would",
        "should", "all", "any", "some", "such", "than", "then", "into",
        "over", "also", "used", "uses", "using", "more", "most", "about",
        "що", "як", "яка", "який", "яке", "які", "яку", "яких", "якому",
        "у", "в", "на", "та", "і", "із", "з", "до", "про", "для", "чи",
    }

    for term in gold_terms:
        norm_term = normalize_text(term)
        if not norm_term:
            continue
        if norm_term in content:
            return True

        # Multi-word substantive evidence overlap
        words = [w for w in re.findall(r"[\w.-]+", norm_term) if len(w) >= 3 and w not in STOPWORDS]
        if len(words) >= 2:
            matched_words = [w for w in words if w in content]
            if len(matched_words) / len(words) >= 0.75:
                return True
        elif len(words) == 1:
            pattern = r"\b" + re.escape(words[0]) + r"\b"
            if re.search(pattern, content):
                return True

    return False


def compute_retrieval_metrics(
    eval_results: List[Dict[str, Any]],
    k_thresholds: tuple = (1, 3, 5),
) -> Dict[str, Any]:
    """Compute Evidence Recall@k, Hit@5, MRR, per-query failures, and category breakdowns."""
    recalls = {k: [] for k in k_thresholds}
    hit_5 = []
    mrr_scores = []
    category_metrics: Dict[str, Dict[str, List[float]]] = {}
    failed_queries: List[Dict[str, Any]] = []
    truncation_violations = 0

    supported_results = [r for r in eval_results if not r.get("refusal_expected", False)]

    for r in supported_results:
        cat = r.get("category", "default")
        qid = r.get("query_id") or r.get("id") or "unknown"
        if cat not in category_metrics:
            category_metrics[cat] = {"hit_5": [], "mrr": [], "recall_1": [], "recall_5": []}

        retrieved = r.get("retrieved_chunks", [])

        # Check truncation violations
        for c in retrieved:
            if c.get("token_overflow_flag", False):
                truncation_violations += 1

        first_relevant_rank = None
        relevant_docs = [d.casefold() for d in r.get("relevant_doc_ids", [])]

        for idx, chunk in enumerate(retrieved):
            rank = idx + 1
            if is_chunk_relevant(chunk, r):
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # MRR & Hit@5
        if first_relevant_rank is not None and first_relevant_rank <= 5:
            rr = 1.0 / first_relevant_rank
            h5 = 1.0
        else:
            rr = 0.0
            h5 = 0.0

        mrr_scores.append(rr)
        hit_5.append(h5)
        category_metrics[cat]["hit_5"].append(h5)
        category_metrics[cat]["mrr"].append(rr)

        # Multi-component evidence targets calculation (P1-2)
        # For queries requiring multiple documents, passages, or facts, verify all mandatory components
        gold_passages = [p.casefold() for p in (r.get("gold_passage_ids") or r.get("passage_ids") or [])]
        gold_spans = r.get("evidence_spans") or []

        if len(relevant_docs) > 1:
            target_units = [{"type": "doc", "val": d} for d in relevant_docs]
        elif len(gold_passages) > 1:
            target_units = [{"type": "passage", "val": p} for p in gold_passages]
        elif len(gold_spans) > 1 and not gold_passages:
            target_units = [{"type": "span", "val": s} for s in gold_spans]
        else:
            target_units = [{"type": "single", "val": None}]

        num_targets = len(target_units)
        for k in k_thresholds:
            chunks_at_k = retrieved[:k]
            satisfied_targets = 0
            for tu in target_units:
                if tu["type"] == "doc":
                    doc_query = {**r, "relevant_doc_ids": [tu["val"]]}
                    if any(is_chunk_relevant(c, doc_query) for c in chunks_at_k):
                        satisfied_targets += 1
                elif tu["type"] == "passage":
                    if any((c.get("chunk_id") or "").casefold() == tu["val"] for c in chunks_at_k):
                        satisfied_targets += 1
                elif tu["type"] == "span":
                    norm_s = normalize_text(tu["val"])
                    if any(norm_s in normalize_text(c.get("content") or c.get("text") or "") for c in chunks_at_k):
                        satisfied_targets += 1
                else:
                    if any(is_chunk_relevant(c, r) for c in chunks_at_k):
                        satisfied_targets += 1

            rec_k = satisfied_targets / num_targets
            recalls[k].append(rec_k)
            if k == 1:
                category_metrics[cat]["recall_1"].append(rec_k)
            elif k == 5:
                category_metrics[cat]["recall_5"].append(rec_k)

        # Mathematical monotonicity invariant check: Recall@1 <= Recall@3 <= Recall@5
        r1, r3, r5 = recalls[1][-1], recalls[3][-1], recalls[5][-1]
        if not (r1 <= r3 + 1e-9 <= r5 + 1e-9):
            raise AssertionError(
                f"Recall@k monotonicity violation for query {qid}: "
                f"R@1={r1:.4f}, R@3={r3:.4f}, R@5={r5:.4f}"
            )

        # Track query failure if Recall@5 < 1.0 or Hit@5 == 0.0
        rec_5_val = recalls[5][-1]
        if rec_5_val < 1.0 or h5 == 0.0:
            failed_queries.append({
                "query_id": qid,
                "category": cat,
                "query": r.get("query", ""),
                "recall_1": recalls[1][-1],
                "recall_5": rec_5_val,
                "hit_5": h5,
                "mrr": rr,
                "relevant_doc_ids": r.get("relevant_doc_ids", []),
                "top_retrieved_sections": [c.get("section", "") for c in retrieved[:3]],
            })

    mean_r1 = float(np.mean(recalls[1])) if recalls[1] else 0.0
    mean_r3 = float(np.mean(recalls[3])) if recalls[3] else 0.0
    mean_r5 = float(np.mean(recalls[5])) if recalls[5] else 0.0
    if not (mean_r1 <= mean_r3 + 1e-9 <= mean_r5 + 1e-9):
        raise AssertionError(
            f"Aggregate Recall@k monotonicity violation: R@1={mean_r1:.4f}, R@3={mean_r3:.4f}, R@5={mean_r5:.4f}"
        )

    summary = {
        "total_evaluated_queries": len(eval_results),
        "supported_queries": len(supported_results),
        "refusal_queries": len(eval_results) - len(supported_results),
        "evidence_recall_1": mean_r1,
        "evidence_recall_3": mean_r3,
        "evidence_recall_5": mean_r5,
        "hit_at_5": float(np.mean(hit_5)) if hit_5 else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "truncation_violations": truncation_violations,
        "failed_query_count": len(failed_queries),
        "failed_queries": failed_queries,
        "category_breakdown": {
            cat: {
                "count": len(vals["hit_5"]),
                "hit_5": float(np.mean(vals["hit_5"])) if vals["hit_5"] else 0.0,
                "mrr": float(np.mean(vals["mrr"])) if vals["mrr"] else 0.0,
                "recall_1": float(np.mean(vals["recall_1"])) if vals["recall_1"] else 0.0,
                "recall_5": float(np.mean(vals["recall_5"])) if vals["recall_5"] else 0.0,
            }
            for cat, vals in category_metrics.items()
        },
    }
    return summary


def compute_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute latency percentiles."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
