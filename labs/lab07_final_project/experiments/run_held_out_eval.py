import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from labs.lab07_final_project.src.config import Settings
from labs.lab07_final_project.src.knowledge_base import KnowledgeBase

queries_path = Path("labs/lab07_final_project/experiments/held_out_retrieval_queries.json")
queries = json.loads(queries_path.read_text(encoding="utf-8"))

settings = Settings()
kb = KnowledgeBase(settings)
kb.initialize()

# Warmup run to stabilize latency and threadpool
for _ in range(2):
    kb.search("warmup query", k=settings.top_k, mode="hybrid")
    kb.search("warmup query", k=settings.top_k, mode="dense")
    kb.search("warmup query", k=settings.top_k, mode="lexical")

out_dir = Path("labs/lab07_final_project/artifacts/portfolio/held_out_eval")
out_dir.mkdir(parents=True, exist_ok=True)

modes = ["hybrid", "dense", "lexical"]
comparison_table = []

for mode in modes:
    print(f"\nEvaluating mode: {mode} on {len(queries)} held-out queries...")
    rows, summary = [], []
    runtimes = []
    
    for q in queries:
        start = time.perf_counter()
        chunks = kb.search(q["query"], k=settings.top_k, mode=mode)
        elapsed_ms = (time.perf_counter() - start) * 1000
        runtimes.append(elapsed_ms)
        
        found = {c.source_id for c in chunks}
        expected = set(q["expected_source_ids"])
        passages = [c.text.casefold() for c in chunks]
        expected_evidence = q.get("expected_evidence", [])
        
        evidence_ranks = [
            next((rank for rank, text in enumerate(passages, 1) if ev.casefold() in text), None)
            for ev in expected_evidence
        ]
        
        hit_at_k = bool(found & expected) if expected else None
        all_hit = (expected <= found) if expected else None
        
        if expected_evidence:
            n_ev = len(expected_evidence)
            q_rec_1 = sum(r is not None and r <= 1 for r in evidence_ranks) / n_ev
            q_rec_3 = sum(r is not None and r <= 3 for r in evidence_ranks) / n_ev
            q_rec_5 = sum(r is not None and r <= 5 for r in evidence_ranks) / n_ev
            q_hit_1 = any(r is not None and r <= 1 for r in evidence_ranks)
            q_hit_3 = any(r is not None and r <= 3 for r in evidence_ranks)
            q_hit_5 = any(r is not None and r <= 5 for r in evidence_ranks)
            # Monotonicity invariant for single query
            assert q_rec_1 <= q_rec_3 <= q_rec_5, f"Monotonicity violation in query {q['query_id']}: {q_rec_1} > {q_rec_3} or {q_rec_3} > {q_rec_5}"
            assert q_hit_1 <= q_hit_3 <= q_hit_5, f"Hit monotonicity violation in query {q['query_id']}"
        else:
            q_rec_1 = q_rec_3 = q_rec_5 = None
            q_hit_1 = q_hit_3 = q_hit_5 = None
        
        summary.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "question_type": q.get("question_type"),
            "hit_at_k": hit_at_k,
            "all_sources_hit": all_hit,
            "evidence_ranks": evidence_ranks,
            "recall_at_1": q_rec_1,
            "recall_at_3": q_rec_3,
            "recall_at_5": q_rec_5,
            "hit_at_1": q_hit_1,
            "hit_at_3": q_hit_3,
            "hit_at_5": q_hit_5,
            "runtime_ms": elapsed_ms
        })
        
        for rank, c in enumerate(chunks, 1):
            rows.append({
                "query_id": q["query_id"],
                "rank": rank,
                "source": c.source_id,
                "section": c.section,
                "chunk_id": c.chunk_id,
                "score_or_distance": c.cosine_distance,
                "text_preview": c.text[:150]
            })
            
    pd.DataFrame(rows).to_csv(out_dir / f"{mode}_results.csv", index=False)
    
    eligible = [x for x in summary if x["hit_at_k"] is not None]
    judged = [x for x in summary if x["recall_at_5"] is not None]
    
    reciprocal_ranks = [
        1.0 / min(r for r in x["evidence_ranks"] if r is not None)
        if any(r is not None for r in x["evidence_ranks"]) else 0.0
        for x in judged
    ]
    
    mean_rec_1 = sum(x["recall_at_1"] for x in judged) / len(judged)
    mean_rec_3 = sum(x["recall_at_3"] for x in judged) / len(judged)
    mean_rec_5 = sum(x["recall_at_5"] for x in judged) / len(judged)
    
    mean_hit_1 = sum(x["hit_at_1"] for x in judged) / len(judged)
    mean_hit_3 = sum(x["hit_at_3"] for x in judged) / len(judged)
    mean_hit_5 = sum(x["hit_at_5"] for x in judged) / len(judged)
    
    assert mean_rec_1 <= mean_rec_3 <= mean_rec_5, f"Macro recall monotonicity violated: {mean_rec_1} <= {mean_rec_3} <= {mean_rec_5}"
    assert mean_hit_1 <= mean_hit_3 <= mean_hit_5, f"Macro hit monotonicity violated"
    
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_latency = sum(runtimes) / len(runtimes)
    
    metrics = {
        "mode": mode,
        "query_count": len(queries),
        "evaluated_queries": len(judged),
        "unsupported_queries": len(queries) - len(judged),
        "source_hit_at_5": sum(x["hit_at_k"] for x in eligible) / len(eligible) if eligible else 0.0,
        "evidence_recall_at_1": round(mean_rec_1, 4),
        "evidence_recall_at_3": round(mean_rec_3, 4),
        "evidence_recall_at_5": round(mean_rec_5, 4),
        "evidence_hit_at_1": round(mean_hit_1, 4),
        "evidence_hit_at_3": round(mean_hit_3, 4),
        "evidence_hit_at_5": round(mean_hit_5, 4),
        "evidence_mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "queries": summary
    }
    
    (out_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    
    comparison_table.append({
        "Configuration": mode,
        "Recall@1": metrics["evidence_recall_at_1"],
        "Recall@3": metrics["evidence_recall_at_3"],
        "Recall@5": metrics["evidence_recall_at_5"],
        "Hit@5": metrics["evidence_hit_at_5"],
        "MRR": metrics["evidence_mrr"],
        "Avg Latency (ms)": metrics["avg_latency_ms"]
    })

print("\n=== HELD-OUT RETRIEVAL EVALUATION RESULTS ===")
print(pd.DataFrame(comparison_table).to_string(index=False))
