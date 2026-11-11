"""Duplicate Audit Tool for Evaluation Datasets.

Compares candidate dataset against all previous datasets:
- frozen_eval_100.json
- dev_eval_50.json
- acceptance_eval_50.json
- final_acceptance_holdout_50.json
- final_acceptance_holdout_v2_50.json

Verifies:
1. Zero exact string matches (normalized lower, whitespace, punctuation).
2. Zero high-similarity near-duplicates (Jaccard word set > 0.60, SequenceMatcher ratio > 0.70).
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

DATASETS_DIR = Path("labs/lab07_final_project/evaluation/datasets")

PREVIOUS_DATASET_FILES = [
    "frozen_eval_100.json",
    "dev_eval_50.json",
    "acceptance_eval_50.json",
    "final_acceptance_holdout_50.json",
    "final_acceptance_holdout_v2_50.json",
    "final_acceptance_holdout_v3_50.json",
    "final_acceptance_holdout_v3_independent_50.json",
]


def normalize_query_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def get_word_set(text: str) -> set[str]:
    return set(normalize_query_text(text).split())


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def audit_against_previous(
    candidate_queries: list[dict],
    jaccard_threshold: float = 0.60,
    seq_threshold: float = 0.70,
) -> tuple[bool, list[dict]]:
    """Audit candidate queries against all previous datasets."""
    previous_queries = []
    for fname in PREVIOUS_DATASET_FILES:
        fpath = DATASETS_DIR / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for item in data:
            previous_queries.append({
                "source_file": fname,
                "id": item["id"],
                "query": item["query"],
                "norm": normalize_query_text(item["query"]),
                "words": get_word_set(item["query"]),
            })

    flagged_issues = []

    # 1. Internal uniqueness check among candidate queries
    seen_cand = {}
    for c in candidate_queries:
        cid = c["id"]
        c_norm = normalize_query_text(c["query"])
        c_words = get_word_set(c["query"])
        if c_norm in seen_cand:
            flagged_issues.append({
                "type": "INTERNAL_EXACT_DUPLICATE",
                "candidate_id": cid,
                "matches_id": seen_cand[c_norm],
                "query": c["query"],
            })
        seen_cand[c_norm] = cid

    # 2. Check against previous queries
    for c in candidate_queries:
        cid = c["id"]
        c_raw = c["query"]
        c_norm = normalize_query_text(c_raw)
        c_words = get_word_set(c_raw)

        for p in previous_queries:
            # Exact normalized match
            if c_norm == p["norm"]:
                flagged_issues.append({
                    "type": "EXACT_DUPLICATE_OF_PREVIOUS",
                    "candidate_id": cid,
                    "previous_source": p["source_file"],
                    "previous_id": p["id"],
                    "candidate_query": c_raw,
                    "previous_query": p["query"],
                })
                continue

            # Jaccard similarity
            jacc = jaccard_similarity(c_words, p["words"])
            seq = sequence_similarity(c_norm, p["norm"])

            if jacc >= jaccard_threshold or seq >= seq_threshold:
                flagged_issues.append({
                    "type": "NEAR_DUPLICATE",
                    "candidate_id": cid,
                    "previous_source": p["source_file"],
                    "previous_id": p["id"],
                    "jaccard": round(jacc, 4),
                    "seq_ratio": round(seq, 4),
                    "candidate_query": c_raw,
                    "previous_query": p["query"],
                })

    is_clean = len(flagged_issues) == 0
    return is_clean, flagged_issues


if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else "final_acceptance_holdout_v3_50.json"
    p = DATASETS_DIR / target_file
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    cand = json.loads(p.read_text(encoding="utf-8"))
    clean, issues = audit_against_previous(cand)
    print(f"Audit of {target_file}: {'PASSED (Zero duplicates)' if clean else 'FAILED'}")
    print(f"Total issues flagged: {len(issues)}")
    for iss in issues[:10]:
        print(json.dumps(iss, indent=2))
    if not clean:
        sys.exit(1)
