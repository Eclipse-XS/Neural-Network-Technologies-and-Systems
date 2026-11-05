"""Execution script for the 100-Query Frozen Scientific Benchmark."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.evaluation.metrics import (
    compute_retrieval_metrics,
    compute_latency_metrics,
)
from labs.lab07_final_project.src.config import Settings
from labs.lab07_final_project.src.knowledge_base import KnowledgeBase
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings


def load_dataset(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def ensure_corpus_ingested(ingestion_service: IngestionService, corpus_dir: Path):
    """Ingest both academic lab reports and engineering documents."""
    supported_exts = {".md", ".txt", ".pdf"}
    files_to_ingest = []
    
    # Root knowledge dir
    for f in corpus_dir.glob("*"):
        if f.is_file() and f.suffix.lower() in supported_exts and f.name != "manifest.json":
            files_to_ingest.append(f)
            
    # Subdirectories (e.g. engineering)
    for f in corpus_dir.glob("**/*"):
        if f.is_file() and f.suffix.lower() in supported_exts and f not in files_to_ingest:
            files_to_ingest.append(f)

    print(f"Discovered {len(files_to_ingest)} documents in {corpus_dir}")
    for file_path in files_to_ingest:
        content = file_path.read_bytes()
        ingestion_service.ingest_file(
            filename=file_path.name,
            content=content,
            document_id=file_path.stem,
        )
    print("Corpus ingestion complete.")


def run_retrieval_benchmark(
    dataset: list[dict],
    query_service: QueryService,
    top_k: int = 5,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> tuple[dict, list[dict]]:
    results = []
    latencies = []

    print(f"Running retrieval benchmark across {len(dataset)} queries (mode={mode.value}, top_k={top_k})...")

    for idx, item in enumerate(dataset, 1):
        query = item["query"]
        t0 = time.perf_counter()
        
        evidence_chunks = query_service.search(
            query=query,
            k=top_k,
            mode=mode,
            filter_quoted_outputs=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_data = []
        for rank, ev in enumerate(evidence_chunks, 1):
            chunk = ev.chunk
            retrieved_data.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.metadata.get("filename", ""),
                "source_id": chunk.metadata.get("source_id", ""),
                "section": chunk.section,
                "content": chunk.text,
                "score_type": ev.score.score_type.value,
                "score_value": ev.score.score_value,
                "token_overflow_flag": chunk.token_count > 240,
            })

        results.append({
            **item,
            "query_id": item.get("query_id") or item.get("id"),
            "category": item["category"],
            "language": item["language"],
            "query": query,
            "refusal_expected": item.get("refusal_expected", False),
            "relevant_doc_ids": item.get("relevant_doc_ids", []),
            "target_sections": item.get("target_sections", []),
            "evidence_spans": item.get("evidence_spans", []),
            "gold_passage_ids": item.get("gold_passage_ids", []),
            "gold_chunks": item.get("gold_chunks", []),
            "retrieved_chunks": retrieved_data,
            "latency_ms": elapsed_ms,
        })

        if idx % 20 == 0 or idx == len(dataset):
            print(f"  Processed {idx}/{len(dataset)} queries...")

    retrieval_metrics = compute_retrieval_metrics(results, k_thresholds=(1, 3, 5))
    latency_metrics = compute_latency_metrics(latencies)

    summary = {
        **retrieval_metrics,
        "latency": latency_metrics,
        "benchmark_config": {
            "mode": mode.value,
            "top_k": top_k,
            "queries_count": len(dataset),
        }
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Knowledge Copilot Frozen Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="labs/lab07_final_project/evaluation/datasets/frozen_eval_100.json")
    parser.add_argument("--corpus", type=str, default="labs/lab07_final_project/assets/knowledge")
    parser.add_argument("--output", type=str, default="labs/lab07_final_project/evaluation/baselines/benchmark_results.json")
    parser.add_argument("--db-path", type=str, default="labs/lab07_final_project/outputs/benchmark.db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["dense", "lexical", "hybrid"])
    parser.add_argument("--eval-all-modes", action="store_true", default=True, help="Evaluate Dense, Lexical, and Hybrid on identical queries")
    parser.add_argument("--fail-under", action="store_true", default=False, help="Exit with code 1 if any target threshold is not met")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db_path)
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    print("=== INITIALIZING BENCHMARK ENVIRONMENT ===")
    dataset = load_dataset(dataset_path)

    # Initialize infrastructure
    settings = Settings()
    sqlite_repo = SQLiteRepository(args.db_path)
    embeddings = CopilotEmbeddings(
        model_name=settings.embedding_model,
        revision=settings.embedding_revision,
        device="cpu",
        local_files_only=True,
    )
    redis_adapter = RedisVectorAdapter(
        redis_url=settings.redis_url,
        index_name="lab07:benchmark:idx",
        key_prefix="lab07:bench:chunk",
    )
    try:
        client = redis_adapter.get_client()
        bench_keys = list(client.scan_iter(match="lab07:bench:*"))
        if bench_keys:
            client.delete(*bench_keys)
    except Exception:
        pass

    redis_adapter.initialize_store(embeddings)

    ingestion_service = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=redis_adapter,
        embeddings_model=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_tokens=settings.embedding_max_tokens,
    )

    ensure_corpus_ingested(ingestion_service, corpus_path)

    query_service = QueryService(
        ingestion_service=ingestion_service,
        redis_adapter=redis_adapter,
        sqlite_repo=sqlite_repo,
        default_retrieval_mode=RetrievalMode(args.mode),
        top_k=args.top_k,
    )

    all_summaries = {}
    itemized_hybrid = []

    if args.eval_all_modes:
        print("\n--- Running Multi-Mode Evaluation: Dense, Lexical, Hybrid ---")
        for m in [RetrievalMode.DENSE, RetrievalMode.LEXICAL, RetrievalMode.HYBRID]:
            print(f"\n>> Evaluating mode: {m.value.upper()}")
            m_summary, m_items = run_retrieval_benchmark(
                dataset=dataset,
                query_service=query_service,
                top_k=args.top_k,
                mode=m,
            )
            all_summaries[m.value] = m_summary
            if m == RetrievalMode.HYBRID:
                itemized_hybrid = m_items
        summary = all_summaries["hybrid"]
        itemized_results = itemized_hybrid
    else:
        mode_enum = RetrievalMode(args.mode)
        summary, itemized_results = run_retrieval_benchmark(
            dataset=dataset,
            query_service=query_service,
            top_k=args.top_k,
            mode=mode_enum,
        )
        all_summaries[args.mode] = summary

    # Validate against target rubrics on Hybrid (primary mode)
    targets = {
        "evidence_recall_1": (0.65, summary["evidence_recall_1"]),
        "evidence_recall_3": (0.80, summary["evidence_recall_3"]),
        "evidence_recall_5": (0.85, summary["evidence_recall_5"]),
        "hit_at_5": (0.92, summary["hit_at_5"]),
        "mrr": (0.75, summary["mrr"]),
    }

    if args.eval_all_modes:
        print("\n" + "=" * 76)
        print("                 RETRIEVAL COMPARISON MATRIX (100 QUERIES)")
        print("=" * 76)
        header = f"{'Metric':<20} {'Target':<10} {'Dense':<14} {'BM25 (Lexical)':<16} {'Hybrid':<14}"
        print(header)
        print("-" * 76)
        dense_s = all_summaries.get("dense", {})
        lex_s = all_summaries.get("lexical", {})
        hyb_s = all_summaries.get("hybrid", {})
        
        for metric, (threshold, _) in targets.items():
            d_val = dense_s.get(metric, 0.0)
            l_val = lex_s.get(metric, 0.0)
            h_val = hyb_s.get(metric, 0.0)
            print(f"{metric:<20} >={threshold:<8.2f} {d_val:<14.4f} {l_val:<16.4f} {h_val:<14.4f}")
        print("-" * 76)
        d_p50 = dense_s.get("latency", {}).get("p50_ms", 0.0)
        l_p50 = lex_s.get("latency", {}).get("p50_ms", 0.0)
        h_p50 = hyb_s.get("latency", {}).get("p50_ms", 0.0)
        print(f"{'Latency P50 (ms)':<20} {'-':<10} {d_p50:<14.2f} {l_p50:<16.2f} {h_p50:<14.2f}")
        d_p95 = dense_s.get("latency", {}).get("p95_ms", 0.0)
        l_p95 = lex_s.get("latency", {}).get("p95_ms", 0.0)
        h_p95 = hyb_s.get("latency", {}).get("p95_ms", 0.0)
        print(f"{'Latency P95 (ms)':<20} {'-':<10} {d_p95:<14.2f} {l_p95:<16.2f} {h_p95:<14.2f}")
        print("=" * 76)

    print("\n" + "=" * 55)
    print("           PRIMARY HYBRID BENCHMARK RESULTS")
    print("=" * 55)
    print(f"Total Evaluated Queries: {summary['total_evaluated_queries']}")
    print(f"Supported Queries:       {summary['supported_queries']}")
    print("-" * 55)
    all_targets_passed = True
    for metric, (threshold, observed) in targets.items():
        status = "PASS" if observed >= threshold else "FAIL"
        if status == "FAIL":
            all_targets_passed = False
        print(f"{metric:<20} Observed: {observed:.4f} (Target >= {threshold:.2f}) -> [{status}]")
    
    print(f"Truncation Violations: {summary['truncation_violations']} (Target == 0) -> [{'PASS' if summary['truncation_violations'] == 0 else 'FAIL'}]")
    print("-" * 55)
    print(f"Latency P50: {summary['latency']['p50_ms']:.2f} ms | P95: {summary['latency']['p95_ms']:.2f} ms | Mean: {summary['latency']['mean_ms']:.2f} ms")
    print("=" * 55)

    print("\nCategory Breakdown (Hybrid):")
    for cat, metrics in summary["category_breakdown"].items():
        print(f"  [{cat.upper()}] (n={metrics['count']}):")
        print(f"    Hit@5: {metrics['hit_5']:.4f} | MRR: {metrics['mrr']:.4f} | Recall@1: {metrics['recall_1']:.4f} | Recall@5: {metrics['recall_5']:.4f}")

    # Output artifact containing all modes
    final_output = {
        "modes": all_summaries,
        "summary": summary,
        "targets_passed": all_targets_passed,
        "itemized_results": itemized_results,
    }
    output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full multi-mode benchmark results to {output_path}")

    if args.fail_under and (not all_targets_passed or summary["truncation_violations"] > 0):
        print("\nWARNING: Some benchmark targets were not met. Inspect category breakdowns.")
        sys.exit(1)
    else:
        print("\nSUCCESS: Multi-mode benchmark execution complete and recorded!")


if __name__ == "__main__":
    main()
