"""Scientific Evaluation Dataset Validator and Audit Tool.

Audits and verifies:
1. Two-tier evaluation dataset integrity and SHA256 cryptographic provenance against manifest.
2. Production chunking consistency: Every gold passage ID exists in the chunked corpus.
3. Verbatim existence of every evidence span in source documents and inside gold passage chunks.
4. Cross-document completeness: Every required document has validated supporting evidence.
5. Strict refusal query schema hygiene.

Read-only by default (suitable for CI). Pass --update-manifest to regenerate the integrity manifest.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.src.config import Settings
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings
from labs.lab07_final_project.src.knowledge_copilot.rag.splitters import build_chunks_from_document
from labs.lab07_final_project.src.knowledge_copilot.rag.extractors import validate_file_content, extract_text_from_bytes
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Document, DocumentStatus
from labs.lab07_final_project.evaluation.metrics import normalize_text

KNOWLEDGE_DIR = Path("labs/lab07_final_project/assets/knowledge")
DATASETS_DIR = Path("labs/lab07_final_project/evaluation/datasets")
MANIFEST_PATH = DATASETS_DIR / "dataset_integrity_manifest.json"
CORPUS_CHUNKS_PATH = DATASETS_DIR / "corpus_chunks_manifest.json"

DATASET_FILES = [
    "frozen_eval_100.json",
    "dev_eval_50.json",
    "acceptance_eval_50.json",
    "final_acceptance_holdout_50.json",
    "final_acceptance_holdout_v2_50.json",
    "final_acceptance_holdout_v3_50.json",
    "final_acceptance_holdout_v3_independent_50.json",
]


def load_source_documents() -> dict[str, str]:
    docs = {}
    for f in sorted(KNOWLEDGE_DIR.glob("**/*.md")):
        docs[f.name] = f.read_text(encoding="utf-8").replace("\r\n", "\n")
        docs[f.stem] = docs[f.name]
    return docs


def generate_production_chunks(
    settings: Settings,
    embeddings: CopilotEmbeddings,
    allow_cached_manifest: bool = False,
) -> tuple[dict, dict, str]:
    chunks_by_id = {}
    chunks_by_doc = {}

    if allow_cached_manifest and CORPUS_CHUNKS_PATH.exists():
        chunk_source = "CACHED_MANIFEST"
        stored = json.loads(CORPUS_CHUNKS_PATH.read_text(encoding="utf-8"))
        for item in stored:
            c = type("ChunkProxy", (), item)()
            chunks_by_id[c.chunk_id] = c
            chunks_by_doc.setdefault(c.document_id, []).append(c)
            chunks_by_doc.setdefault(c.filename, []).append(c)
    elif embeddings.tokenizer is not None:
        chunk_source = "DYNAMIC_TOKENIZER"
        for f in sorted(KNOWLEDGE_DIR.glob("**/*.md")):
            content = f.read_bytes().replace(b"\r\n", b"\n")
            clean_name, sha256_hash, size_bytes = validate_file_content(content, f.name)
            doc = Document(
                document_id=f.stem,
                filename=clean_name,
                content_type="md",
                sha256=sha256_hash,
                size_bytes=size_bytes,
                status=DocumentStatus.PROCESSING,
                chunk_count=0,
                collection="default",
                created_at="",
                updated_at="",
            )
            text = extract_text_from_bytes(content, clean_name)
            chunks = build_chunks_from_document(
                doc=doc,
                text=text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                max_tokens=settings.embedding_max_tokens,
                token_counter=embeddings.count_tokens,
            )
            chunks_by_doc[f.stem] = chunks
            chunks_by_doc[f.name] = chunks
            for c in chunks:
                chunks_by_id[c.chunk_id] = c
    elif not allow_cached_manifest and embeddings.tokenizer is None:
        raise RuntimeError(
            "Production tokenizer is unavailable and --allow-cached-manifest was not provided. "
            "Default verification mode enforces dynamic chunk regeneration via HuggingFace tokenizer. "
            "Pass --allow-cached-manifest for offline regression audit."
        )
    else:
        raise RuntimeError("No tokenizer available and corpus_chunks_manifest.json is missing.")

    return chunks_by_id, chunks_by_doc, chunk_source


def validate_dataset_file(
    filename: str,
    chunks_by_id: dict,
    chunks_by_doc: dict,
    docs: dict,
    trusted_manifest: dict = None,
) -> dict:
    file_path = DATASETS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file missing: {file_path}")

    text_content = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    sha256_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    data = json.loads(text_content)

    # Cryptographic integrity check against trusted manifest
    if trusted_manifest and filename in trusted_manifest:
        expected_hash = trusted_manifest[filename].get("sha256")
        if expected_hash and sha256_hash != expected_hash:
            raise AssertionError(
                f"Cryptographic provenance mismatch for {filename}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Observed: {sha256_hash}"
            )

    print(f"\n--- Validating {filename} ({len(data)} queries, SHA256: {sha256_hash[:12]}...) ---")

    required_keys = {
        "id", "category", "language", "query", "expected_answer",
        "relevant_doc_ids", "target_sections", "evidence_spans",
        "gold_passage_ids", "refusal_expected"
    }

    supported_count = 0
    refusal_count = 0
    total_spans = 0
    total_passages = 0

    for idx, q in enumerate(data, 1):
        qid = q.get("id", f"idx_{idx}")
        missing_keys = required_keys - set(q.keys())
        if missing_keys:
            raise ValueError(f"Query {qid} missing required keys: {missing_keys}")

        is_refusal = q["refusal_expected"]
        if is_refusal:
            refusal_count += 1
            if q["evidence_spans"]:
                raise AssertionError(f"Refusal query {qid} must have empty evidence_spans")
            if q["gold_passage_ids"]:
                raise AssertionError(f"Refusal query {qid} must have empty gold_passage_ids")
            if q["relevant_doc_ids"]:
                raise AssertionError(f"Refusal query {qid} must have empty relevant_doc_ids")
        else:
            supported_count += 1
            spans = q["evidence_spans"]
            pids = q["gold_passage_ids"]
            rel_docs = [d.replace(".md", "") for d in q["relevant_doc_ids"]]

            if not spans:
                raise AssertionError(f"Supported query {qid} has no evidence_spans")
            if not pids:
                raise AssertionError(f"Supported query {qid} has no gold_passage_ids")
            if not rel_docs:
                raise AssertionError(f"Supported query {qid} has no relevant_doc_ids")

            # 1. Verify gold passage IDs exist in production chunks
            for pid in pids:
                total_passages += 1
                if pid not in chunks_by_id:
                    raise AssertionError(f"Query {qid} gold_passage_id {pid} does not exist in production chunks!")

            # 2. Verify verbatim evidence spans in source documents
            for span in spans:
                total_spans += 1
                found_in_doc = False
                for rd in rel_docs:
                    if rd in docs and span in docs[rd]:
                        found_in_doc = True
                        break
                if not found_in_doc:
                    raise AssertionError(f"Query {qid} evidence span not found in referenced documents {rel_docs}:\n  {span[:100]}")

            # 3. Verify consistency: every gold chunk contains at least one evidence span
            for pid in pids:
                chunk = chunks_by_id[pid]
                c_norm = normalize_text(chunk.text)
                has_span = any(normalize_text(span) in c_norm for span in spans)
                if not has_span:
                    raise AssertionError(
                        f"Consistency error in query {qid}: chunk {pid} in {chunk.document_id} does not contain any evidence span!"
                    )

            # 4. Cross-document completeness: every required document must have supporting evidence
            for rd in rel_docs:
                doc_has_pid = any(chunks_by_id[pid].document_id == rd for pid in pids)
                if not doc_has_pid:
                    raise AssertionError(
                        f"Cross-document error in query {qid}: required document {rd} has no corresponding gold_passage_ids!"
                    )

    print(f"  [PASS] {filename}: {len(data)} queries ({supported_count} supported, {refusal_count} refusal).")
    print(f"  [PASS] {total_spans} verbatim spans & {total_passages} passage IDs verified against production chunks.")

    return {
        "filename": filename,
        "query_count": len(data),
        "supported_queries": supported_count,
        "refusal_queries": refusal_count,
        "verified_evidence_spans": total_spans,
        "verified_gold_passages": total_passages,
        "sha256": sha256_hash,
        "status": "VALIDATED",
    }


def main():
    parser = argparse.ArgumentParser(description="Scientific Evaluation Dataset Audit and Verification Tool")
    parser.add_argument("--update-manifest", action="store_true", help="Update dataset_integrity_manifest.json with new hashes")
    parser.add_argument("--allow-cached-manifest", action="store_true", help="Allow fallback to cached corpus_chunks_manifest.json when tokenizer is missing")
    args = parser.parse_args()

    print("==================================================")
    print(" SCIENTIFIC EVALUATION DATASET AUDIT AND VERIFIER")
    print("==================================================")

    trusted_manifest = {}
    if MANIFEST_PATH.exists() and not args.update_manifest:
        trusted_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        print(f"Loaded trusted manifest from {MANIFEST_PATH} ({len(trusted_manifest)} files registered)")

    settings = Settings()
    embeddings = CopilotEmbeddings(
        model_name=settings.embedding_model,
        revision=settings.embedding_revision,
        device="cpu",
        local_files_only=True,
    )
    docs = load_source_documents()
    chunks_by_id, chunks_by_doc, chunk_source = generate_production_chunks(
        settings, embeddings, allow_cached_manifest=args.allow_cached_manifest
    )
    print(f"Chunk Verification Source: {chunk_source}")
    print(f"Generated/Loaded {len(chunks_by_id)} production corpus chunks across {len(docs)//2} documents.")

    manifest_data = {}
    for fname in DATASET_FILES:
        info = validate_dataset_file(
            filename=fname,
            chunks_by_id=chunks_by_id,
            chunks_by_doc=chunks_by_doc,
            docs=docs,
            trusted_manifest=trusted_manifest,
        )
        manifest_data[fname] = info

    if args.update_manifest:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"\n[UPDATED] Integrity manifest written to {MANIFEST_PATH}")
    else:
        print(f"\n[VERIFIED] Read-only audit passed successfully. Manifest untouched.")


if __name__ == "__main__":
    main()
