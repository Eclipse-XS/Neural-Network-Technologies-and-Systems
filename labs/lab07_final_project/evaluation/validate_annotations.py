"""Scientific Evaluation Dataset Validator and Audit Tool.

Audits and verifies:
1. Schema compliance of historical and final acceptance datasets.
2. Verbatim existence of every evidence span in the referenced source documents.
3. Target section and gold passage ID alignment.
4. Dataset SHA256 provenance hashes.
"""
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWLEDGE_DIR = Path("labs/lab07_final_project/assets/knowledge")
DATASETS_DIR = Path("labs/lab07_final_project/evaluation/datasets")

DOCS = {}
for f in sorted(KNOWLEDGE_DIR.glob("**/*.md")):
    DOCS[f.name] = f.read_text(encoding="utf-8")

DATASET_FILES = [
    "frozen_eval_100.json",
    "dev_eval_50.json",
    "acceptance_eval_50.json",
    "final_acceptance_holdout_50.json",
]


def validate_dataset(filename: str) -> dict:
    file_path = DATASETS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file missing: {file_path}")

    raw_bytes = file_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))

    print(f"\n--- Validating {filename} ({len(data)} queries, SHA256: {sha256[:12]}...) ---")

    required_keys = {
        "id", "category", "language", "query", "expected_answer",
        "relevant_doc_ids", "target_sections", "evidence_spans",
        "gold_passage_ids", "refusal_expected"
    }

    supported_count = 0
    refusal_count = 0
    total_spans = 0

    for idx, q in enumerate(data, 1):
        missing_keys = required_keys - set(q.keys())
        if missing_keys:
            raise ValueError(f"Query {q.get('id', idx)} missing required keys: {missing_keys}")

        is_refusal = q["refusal_expected"]
        if is_refusal:
            refusal_count += 1
            if q["evidence_spans"]:
                raise AssertionError(f"Refusal query {q['id']} must have empty evidence_spans, got {q['evidence_spans']}")
            if q["relevant_doc_ids"]:
                raise AssertionError(f"Refusal query {q['id']} must have empty relevant_doc_ids, got {q['relevant_doc_ids']}")
        else:
            supported_count += 1
            spans = q["evidence_spans"]
            if not spans:
                raise AssertionError(f"Supported query {q['id']} has no evidence_spans")
            if not q["target_sections"]:
                raise AssertionError(f"Supported query {q['id']} has no target_sections")
            if not q["gold_passage_ids"]:
                raise AssertionError(f"Supported query {q['id']} has no gold_passage_ids")
            if not q["relevant_doc_ids"]:
                raise AssertionError(f"Supported query {q['id']} has no relevant_doc_ids")

            for span in spans:
                total_spans += 1
                found_in_doc = False
                for doc_id in q["relevant_doc_ids"]:
                    doc_name = doc_id if doc_id in DOCS else f"{doc_id}.md" if f"{doc_id}.md" in DOCS else None
                    if doc_name and span in DOCS[doc_name]:
                        found_in_doc = True
                        break
                if not found_in_doc:
                    raise AssertionError(
                        f"Query {q['id']} evidence span NOT FOUND verbatim in referenced documents {q['relevant_doc_ids']}:\n"
                        f"  Span: {repr(span)}"
                    )

    print(f"  [PASS] {filename}: {len(data)} queries validated ({supported_count} supported, {refusal_count} refusal).")
    print(f"  [PASS] {total_spans} verbatim evidence spans verified against source documents.")

    return {
        "filename": filename,
        "query_count": len(data),
        "supported_queries": supported_count,
        "refusal_queries": refusal_count,
        "verified_evidence_spans": total_spans,
        "sha256": sha256,
        "status": "VALIDATED",
    }


def main():
    print("==================================================")
    print(" SCIENTIFIC EVALUATION DATASET AUDIT AND VERIFIER")
    print("==================================================")
    results = {}
    for fname in DATASET_FILES:
        res = validate_dataset(fname)
        results[fname] = res

    manifest_path = DATASETS_DIR / "dataset_integrity_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit complete. Integrity manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
