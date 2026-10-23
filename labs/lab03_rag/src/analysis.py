"""Offline catalog analysis and evaluation-only helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .documents import number


def load_catalog(directory: Path) -> pd.DataFrame:
    """Load the three frozen CSV files without altering source values."""
    frames = []
    for path in sorted(Path(directory).glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        frame["source_file"] = path.name
        frame["row_id"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    if len(frames) != 3:
        raise ValueError(f"Expected exactly three evaluation CSV files; found {len(frames)}")
    result = pd.concat(frames, ignore_index=True)
    for name in ("initial_price", "final_price", "discount", "rating", "ratings_count"):
        result[f"{name}_numeric"] = result[name].map(number)
    result["rating_observed"] = ~(
        result["rating_numeric"].eq(0) & result["ratings_count_numeric"].eq(0))
    result["description_chars"] = result["product_description"].str.len()
    expected = result["initial_price_numeric"] * (1 - result["discount_numeric"] / 100)
    result["price_formula_delta_inr"] = (result["final_price_numeric"] - expected).abs()
    result["price_conflict"] = result["price_formula_delta_inr"].gt(1) & result["discount_numeric"].notna()
    return result


def load_replay_catalog(path: Path) -> pd.DataFrame:
    """Load the versioned, derived catalog snapshot used only for portfolio replay."""
    frame = pd.read_csv(path, dtype={"product_id": str}, keep_default_na=False, encoding="utf-8-sig")
    for name in ("initial_price", "final_price", "discount", "rating", "ratings_count"):
        frame[f"{name}_numeric"] = frame[name].map(number)
    frame["rating_observed"] = ~(
        frame["rating_numeric"].eq(0) & frame["ratings_count_numeric"].eq(0))
    frame["description_chars"] = frame["product_description"].str.len()
    expected = frame["initial_price_numeric"] * (1 - frame["discount_numeric"] / 100)
    frame["price_formula_delta_inr"] = (frame["final_price_numeric"] - expected).abs()
    frame["price_conflict"] = frame["price_formula_delta_inr"].gt(1) & frame["discount_numeric"].notna()
    return frame


def catalog_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)), "columns_in_source": 24,
        "categories": frame.groupby("category").size().sort_values(ascending=False).to_dict(),
        "missing_discount": int(frame["discount_numeric"].isna().sum()),
        "unrated_rows": int((~frame["rating_observed"]).sum()),
        "price_conflicts": int(frame["price_conflict"].sum()),
        "duplicate_rows": int(frame.drop(columns=["source_file", "row_id"]).duplicated().sum()),
    }


def evaluate_id_lists(queries: Iterable[dict], retrieved: dict[str, list[str]], k: int = 5) -> pd.DataFrame:
    """Evaluation layer: this is the only path that consumes relevance labels."""
    rows = []
    for query in queries:
        expected = set(map(str, query.get("relevant_product_ids", [])))
        actual = list(map(str, retrieved.get(query["id"], [])))[:k]
        ranks = [rank for rank, value in enumerate(actual, 1) if value in expected]
        rows.append({
            "query_id": query["id"], "kind": query.get("kind", "unknown"), "k": k,
            "expected_count": len(expected), "returned": len(actual),
            "hit_at_k": None if not expected else int(bool(ranks)),
            "first_relevant_rank": min(ranks) if ranks else None,
            "mrr": None if not expected else (1 / min(ranks) if ranks else 0.0),
            "recall_at_k": None if not expected else len(expected.intersection(actual)) / len(expected),
            "coverage": None if not expected else len(expected.intersection(actual)) / len(expected),
        })
    return pd.DataFrame(rows)


def load_manual_review(path: Path) -> pd.DataFrame:
    reviews = json.loads(Path(path).read_text(encoding="utf-8"))
    taxonomy = {
        "q01": "reasoning_over_conflicting_fields", "q02": "missing_value_semantics",
        "q03": "unit_error_and_citation_format", "q04": "incomplete_multi_record_retrieval",
        "q05": "semantic_search_used_for_global_aggregation", "q06": "arithmetic_error",
        "q07": "unsupported_claim", "q08": "correct_abstention",
    }
    return pd.DataFrame([{"query_id": key, "failure_category": taxonomy[key], "review": value}
                         for key, value in reviews.items()])
