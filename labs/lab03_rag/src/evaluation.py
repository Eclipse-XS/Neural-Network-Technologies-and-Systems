"""Fixed corpus-derived queries, retrieval metrics and limited lexical evidence."""
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
import pandas as pd
from .config import LAB_ROOT
from .retrieval import retrieve, result_table
from .rag import generate_answer


def load_queries(data_dir, path=LAB_ROOT / "queries.json"):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    actual_files = {p.name for p in Path(data_dir).glob("*.csv")}
    if actual_files != set(manifest["corpus_sha256"]):
        raise ValueError("Fixed evaluation requires the three audited original CSV files")
    for name, expected in manifest["corpus_sha256"].items():
        if hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Corpus changed: {name}. Review ground truth before running evaluation.")
    return manifest["queries"]


def relevance_metrics(retrieved_ids, expected_ids):
    if not expected_ids:
        return dict(hit_at_k=None, first_relevant_rank=None, reciprocal_rank=None, recall_at_k=None)
    expected = set(expected_ids)
    ranks = [rank for rank, rid in enumerate(retrieved_ids, 1) if rid in expected]
    return dict(hit_at_k=int(bool(ranks)), first_relevant_rank=min(ranks) if ranks else None,
                reciprocal_rank=1 / min(ranks) if ranks else 0,
                recall_at_k=len(set(retrieved_ids) & expected) / len(expected))


def run_retrieval(store, queries, output_dir, k=5):
    frames, summaries, results = [], [], {}
    for q in queries:
        pairs, elapsed = retrieve(store, q["question"], k, q.get("category_filter"))
        results[q["id"]] = pairs
        frame = result_table(q["question"], pairs, elapsed)
        frame.insert(0, "query_id", q["id"])
        frames.append(frame)
        summaries.append(dict(query_id=q["id"], query=q["question"], kind=q["kind"],
                              returned=len(pairs), query_elapsed_ms=elapsed,
                              **relevance_metrics([str(d.metadata["product_id"]) for d, _ in pairs],
                                                  q["relevant_product_ids"])))
    table = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    output = Path(output_dir) / "retrieval"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    # Timing includes embedding (possibly cached), network and parsing; not Redis-only latency.
    metrics = dict(hit_at_k=summary.hit_at_k.mean(), mrr=summary.reciprocal_rank.mean(),
                   scored_queries=int(summary.hit_at_k.notna().sum()), k=k,
                   mean_query_ms=summary.query_elapsed_ms.mean())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return table, summary, results


def field_present(answer, value):
    """Boundary-aware lexical check; does not prove attribution or entailment."""
    normalized = answer.casefold().replace(",", "")
    value = str(value).strip().strip('"').replace("₹", "").replace(",", "").strip().casefold()
    if not value:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        value = value.rstrip("0").rstrip(".") if "." in value else value
        pattern = r"(?<![\w.])" + re.escape(value) + r"(?:\.0+)?(?!\w|\.\d)"
        return bool(re.search(pattern, normalized))
    return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", normalized))


def verify_answer(answer, documents, expected_fields):
    allowed = {d.metadata["record_id"] for d in documents}
    citations = set(re.findall(r"\[([^\[\]\n]+\.csv:\d+)\]", answer))
    checks = []
    for expected in expected_fields:
        checks.append(dict(product_id=expected["product_id"], record_id=expected["record_id"],
                           field=expected["field"], expected=expected["value"],
                           value_present=field_present(answer, expected["value"]),
                           expected_citation_present=expected["record_id"] in citations,
                           expected_record_retrieved=expected["record_id"] in allowed))
    return dict(citations=sorted(citations), unknown_citations=sorted(citations - allowed),
                valid_citation_present=bool(citations & allowed), field_checks=checks,
                note="Lexical matches and citation validity only; manual attribution/semantic review required.")


def run_answers(llm, queries, retrieval_results, output_dir, model_id):
    """Reuse exactly the inspected Redis results. Save each completed query, including failures."""
    output = Path(output_dir) / "rag"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in queries:
        docs = [d for d, _ in retrieval_results[q["id"]]]
        started = perf_counter()
        try:
            answer, messages = generate_answer(llm, q["question"], docs)
            verification = verify_answer(answer, docs, q["expected_fields"])
            status, error = "completed", ""
            (output / f"{q['id']}_prompt.json").write_text(
                json.dumps([dict(role=m.type, content=m.content) for m in messages], ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as exc:
            answer, verification = "", {}
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        rows.append(dict(query_id=q["id"], query=q["question"], model=model_id,
                         retrieved_record_ids=json.dumps([d.metadata["record_id"] for d in docs]),
                         expected_fields=json.dumps(q["expected_fields"], ensure_ascii=False),
                         answer=answer, verification=json.dumps(verification, ensure_ascii=False),
                         status=status, error=error, generation_ms=(perf_counter()-started)*1000,
                         manual_review="NOT REVIEWED"))
        pd.DataFrame(rows).to_csv(output / "final_answers.csv", index=False)
    return pd.DataFrame(rows)
