"""Offline analysis and deterministic retrieval helpers for the educational notebook."""
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
    for field in ("initial_price", "final_price", "discount", "rating", "ratings_count"):
        result[f"{field}_numeric"] = result[field].map(number)
    result["rating_observed"] = ~(
        result["rating_numeric"].eq(0) & result["ratings_count_numeric"].eq(0)
    )
    result["description_chars"] = result["product_description"].str.len()
    expected = result["initial_price_numeric"] * (1 - result["discount_numeric"] / 100)
    result["price_formula_delta_inr"] = (result["final_price_numeric"] - expected).abs()
    result["price_conflict"] = result["price_formula_delta_inr"].gt(1) & result["discount_numeric"].notna()
    return result


def catalog_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns_in_source": 24,
        "categories": frame.groupby("category").size().sort_values(ascending=False).to_dict(),
        "missing_discount": int(frame["discount_numeric"].isna().sum()),
        "unrated_rows": int((~frame["rating_observed"]).sum()),
        "price_conflicts": int(frame["price_conflict"].sum()),
        "duplicate_rows": int(frame.drop(columns=["source_file", "row_id"]).duplicated().sum()),
    }


def structured_retrieve(frame: pd.DataFrame, query: dict, k: int = 5) -> pd.DataFrame:
    """Deterministic rules for queries whose intent is numeric or exact-ID based.

    Semantic queries deliberately return an empty frame: this function does not pretend
    that hand-written filters solve free-form relevance.
    """
    kind = query["kind"]
    result = frame.copy()
    ids = [str(x) for x in query.get("relevant_product_ids", [])]
    if kind in {"direct", "comparison", "cross_category"}:
        result = result[result["product_id"].isin(ids)]
        order = {value: i for i, value in enumerate(ids)}
        result = result.assign(_order=result["product_id"].map(order)).sort_values("_order")
    elif query["id"] == "q03":
        result = result[(result["category"] == "earrings") &
                        result["product_description"].str.contains("silver", case=False, na=False)]
        result = result.sort_values("final_price_numeric")
    elif query["id"] == "q05":
        result = result[(result["category"] == "sports-shoes") &
                        result["rating_numeric"].ge(4) &
                        result["discount_numeric"].gt(0)]
        result = result.sort_values(["final_price_numeric", "product_id"])
    else:
        return result.iloc[0:0]
    return result.head(k).drop(columns=["_order"], errors="ignore")


def evaluate_id_lists(queries: Iterable[dict], retrieved: dict[str, list[str]], k: int = 5) -> pd.DataFrame:
    rows = []
    for query in queries:
        expected = set(map(str, query.get("relevant_product_ids", [])))
        actual = list(map(str, retrieved.get(query["id"], [])))[:k]
        ranks = [i for i, value in enumerate(actual, 1) if value in expected]
        rows.append({
            "query_id": query["id"], "kind": query["kind"], "k": k,
            "expected_count": len(expected), "returned": len(actual),
            "hit_at_k": None if not expected else int(bool(ranks)),
            "first_relevant_rank": min(ranks) if ranks else None,
            "mrr": None if not expected else (1 / min(ranks) if ranks else 0.0),
            "recall_at_k": None if not expected else len(expected.intersection(actual)) / len(expected),
            "coverage": None if not expected else len(expected.intersection(actual)) / len(expected),
        })
    return pd.DataFrame(rows)


def numeric_answer(query: dict, rows: pd.DataFrame) -> str:
    """Produce auditable answers for exact/numeric cases without delegating arithmetic to an LLM."""
    if rows.empty:
        return "Structured retrieval is not applicable or returned no rows."
    citations = [f"[{r.source_file}:{int(r.row_id)}]" for r in rows.itertuples()]
    if query["id"] == "q05":
        r = rows.iloc[0]
        return (f"Minimum among all matching rows: {r['title']} (product {r['product_id']}), "
                f"{r['final_price_numeric']:.0f} INR, rating {r['rating_numeric']:.1f}, "
                f"discount {r['discount_numeric']:.0f}% {citations[0]}.")
    parts = []
    for r, citation in zip(rows.itertuples(), citations):
        discount = "not provided" if pd.isna(r.discount_numeric) else f"{r.discount_numeric:.0f}%"
        parts.append(f"{r.title} ({r.product_id}): {r.final_price_numeric:.0f} INR; "
                     f"discount {discount}; rating {r.rating_numeric:.1f} {citation}")
    return "\n".join(parts)


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
