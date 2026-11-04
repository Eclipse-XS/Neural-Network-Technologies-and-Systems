"""Evaluation metrics calculation for RAG retrieval and generation."""
import numpy as np
from typing import Dict, List, Any, Optional


def is_chunk_relevant(chunk: dict, gold_query: dict) -> bool:
    """Determine if a retrieved chunk matches ground truth targets."""
    filename = (chunk.get("filename") or "").casefold()
    source_id = (chunk.get("source_id") or "").casefold()
    document_id = (chunk.get("document_id") or "").casefold()
    content = (chunk.get("content") or chunk.get("text") or "").casefold()
    section = (chunk.get("section") or "").casefold()

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
    else:
        doc_match = True

    gold_terms = [g.casefold() for g in gold_query.get("gold_chunks", [])]
    if not gold_terms:
        return doc_match

    # Clean markdown formatting (backticks, asterisks, underscores) for matching
    import re
    clean_content = re.sub(r'[`*_]', '', content)
    clean_section = re.sub(r'[`*_]', '', section)
    clean_terms = [re.sub(r'[`*_]', '', g) for g in gold_terms]

    # Distinctive evidence span matching with stopword filtering
    STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "has", "have", "had", "not", "but", "what", "which", "who",
        "how", "why", "when", "where", "can", "could", "will", "would",
        "should", "all", "any", "some", "such", "than", "then", "into",
        "over", "also", "used", "uses", "using", "more", "most", "about",
    }

    for term in clean_terms:
        if term in clean_content or term in clean_section:
            return True
        # Multi-word substantive evidence overlap (>= 75% non-stopwords)
        words = [w for w in re.findall(r"[\w.-]+", term) if len(w) >= 3 and w not in STOPWORDS]
        if len(words) >= 2:
            matched_words = [w for w in words if w in clean_content]
            if len(matched_words) / len(words) >= 0.75:
                return True
        elif len(words) == 1:
            # Single distinctive keyword must match as a complete token
            pattern = r"\b" + re.escape(words[0]) + r"\b"
            if re.search(pattern, clean_content):
                return True
    return False


def compute_retrieval_metrics(
    eval_results: List[Dict[str, Any]],
    k_thresholds: tuple = (1, 3, 5),
) -> Dict[str, Any]:
    """Compute Evidence Recall@k, Hit@5, MRR, and category breakdowns.
    
    eval_results: list of dicts with keys:
      - 'query_id': str
      - 'category': str
      - 'refusal_expected': bool
      - 'relevant_doc_ids': list[str]
      - 'gold_chunks': list[str]
      - 'retrieved_chunks': list[dict] (ranked top-k)
      - 'latency_ms': float
    """
    recalls = {k: [] for k in k_thresholds}
    hit_5 = []
    mrr_scores = []
    category_metrics: Dict[str, Dict[str, List[float]]] = {}
    truncation_violations = 0

    supported_results = [r for r in eval_results if not r.get("refusal_expected", False)]

    for r in supported_results:
        cat = r.get("category", "default")
        if cat not in category_metrics:
            category_metrics[cat] = {"hit_5": [], "mrr": [], "recall_1": [], "recall_5": []}

        retrieved = r.get("retrieved_chunks", [])
        
        # Check truncation violations (if content exceeds 1000 chars without budget guard)
        for c in retrieved:
            if c.get("token_overflow_flag", False):
                truncation_violations += 1

        first_relevant_rank = None
        gold_terms = [g.casefold() for g in r.get("gold_chunks", [])]
        relevant_docs = [d.casefold() for d in r.get("relevant_doc_ids", [])]
        
        # Track which gold chunks were retrieved
        matched_gold_terms = set()

        for idx, chunk in enumerate(retrieved):
            rank = idx + 1
            if is_chunk_relevant(chunk, r):
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # MRR
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

        # Recall@k
        for k in k_thresholds:
            chunks_at_k = retrieved[:k]
            if len(relevant_docs) > 1 and k >= len(relevant_docs):
                # Multi-document cross-comparison recall at k >= 2: fraction of required documents retrieved
                retrieved_doc_matches = 0
                for r_doc in relevant_docs:
                    if any(
                        is_chunk_relevant(c, {**r, "relevant_doc_ids": [r_doc]})
                        for c in chunks_at_k
                    ):
                        retrieved_doc_matches += 1
                rec_k = retrieved_doc_matches / len(relevant_docs)
            else:
                # Single document query (or k=1 where only 1 chunk is retrieved): 1.0 if any relevant chunk is in top k, else 0.0
                rec_k = 1.0 if any(is_chunk_relevant(c, r) for c in chunks_at_k) else 0.0

            recalls[k].append(rec_k)
            if k == 1:
                category_metrics[cat]["recall_1"].append(rec_k)
            elif k == 5:
                category_metrics[cat]["recall_5"].append(rec_k)

    summary = {
        "total_evaluated_queries": len(eval_results),
        "supported_queries": len(supported_results),
        "evidence_recall_1": float(np.mean(recalls[1])) if recalls[1] else 0.0,
        "evidence_recall_3": float(np.mean(recalls[3])) if recalls[3] else 0.0,
        "evidence_recall_5": float(np.mean(recalls[5])) if recalls[5] else 0.0,
        "hit_at_5": float(np.mean(hit_5)) if hit_5 else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "truncation_violations": truncation_violations,
        "category_breakdown": {
            cat: {
                "count": len(vals["hit_5"]),
                "hit_5": float(np.mean(vals["hit_5"])) if vals["hit_5"] else 0.0,
                "mrr": float(np.mean(vals["mrr"])) if vals["mrr"] else 0.0,
                "recall_1": float(np.mean(vals["recall_1"])) if vals["recall_1"] else 0.0,
                "recall_5": float(np.mean(vals["recall_5"])) if vals["recall_5"] else 0.0,
            }
            for cat, vals in category_metrics.items()
        }
    }
    return summary


def compute_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute latency percentiles."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
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
