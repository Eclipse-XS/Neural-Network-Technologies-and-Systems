"""Leakage-free intent parsing and hybrid catalog retrieval.

This inference module accepts only user text, catalog rows, and optional semantic
candidates. Evaluation labels and query-case identifiers are intentionally absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Intent:
    strategy: str
    product_ids: tuple[str, ...] = ()
    unknown_product_ids: tuple[str, ...] = ()
    category: str | None = None
    attributes: tuple[str, ...] = ()
    rating_min: float | None = None
    positive_discount: bool = False
    aggregation: str | None = None
    unsupported_reason: str | None = None


@dataclass
class RetrievalResult:
    question: str
    intent: Intent
    rows: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)

    @property
    def product_ids(self) -> list[str]:
        return self.rows["product_id"].astype(str).tolist()


_CATEGORY_ALIASES = {
    "sports-shoes": ("sports shoes", "sport shoes", "sports-shoes", "athletic shoes"),
    "earrings": ("earring", "earrings"),
    "dresses": ("dress", "dresses"),
}
_ID_TOKEN = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")
_RATING_MIN = re.compile(
    r"(?:rating\s*(?:of\s*)?|rated\s*)(?:at\s+least|>=|not\s+less\s+than)\s*(\d(?:\.\d+)?)", re.I)


def parse_intent(question: str, catalog_product_ids: Iterable[str]) -> Intent:
    """Parse a transparent, deliberately limited subset of catalog questions."""
    text = " ".join(str(question).split())
    lowered = text.casefold()
    catalog_ids = set(map(str, catalog_product_ids))
    candidates = list(dict.fromkeys(_ID_TOKEN.findall(text)))
    known_ids = tuple(value for value in candidates if value in catalog_ids)
    unknown_ids = tuple(value for value in candidates if value not in catalog_ids)

    if any(term in lowered for term in ("shipping fee", "delivered to", "delivery to")):
        return Intent("unsupported", known_ids, unknown_ids,
                      unsupported_reason="Destination/date-specific delivery data is absent from the catalog.")
    if known_ids or unknown_ids:
        return Intent("exact_id", known_ids, unknown_ids)

    category = next((name for name, aliases in _CATEGORY_ALIASES.items()
                     if any(alias in lowered for alias in aliases)), None)
    attributes = []
    if "silver-plated" in lowered or "silver plated" in lowered:
        attributes.append("silver-plated")
    for attribute in ("cotton", "maxi", "casual"):
        if re.search(rf"\b{attribute}\b", lowered):
            attributes.append(attribute)
    rating_match = _RATING_MIN.search(text)
    rating_min = float(rating_match.group(1)) if rating_match else None
    positive_discount = bool(re.search(
        r"(?:positive|non[- ]?zero)\s+discount|discount\s*(?:>|above|greater\s+than)\s*0", lowered))
    aggregation = None
    if re.search(r"\b(lowest|minimum|cheapest)\b", lowered):
        aggregation = "min_final_price"
    elif re.search(r"\b(highest|maximum|most expensive)\b", lowered):
        aggregation = "max_final_price"

    if category or attributes or rating_min is not None or positive_discount or aggregation:
        return Intent("structured", category=category, attributes=tuple(attributes),
                      rating_min=rating_min, positive_discount=positive_discount,
                      aggregation=aggregation)
    return Intent("semantic")


def _attribute_mask(frame: pd.DataFrame, attribute: str) -> pd.Series:
    searchable = (frame["product_description"].fillna("") + " " +
                  frame["product_specifications"].fillna("") + " " +
                  frame.get("product_details", pd.Series("", index=frame.index)).fillna(""))
    if attribute == "silver-plated":
        return searchable.str.contains(r"silver[- ]plated", case=False, regex=True)
    if attribute in {"cotton", "maxi", "casual"}:
        return searchable.str.contains(rf"\b{re.escape(attribute)}\b", case=False, regex=True)
    raise ValueError(f"Unsupported attribute rule: {attribute}")


def route_and_retrieve(frame: pd.DataFrame, question: str,
                       semantic_product_ids: Iterable[str] = (), k: int = 5) -> RetrievalResult:
    """Select a strategy and retrieve without accepting evaluation objects or labels."""
    if k < 1:
        raise ValueError("k must be positive")
    intent = parse_intent(question, frame["product_id"])
    result = frame.copy()
    diagnostics = {"strategy": intent.strategy, "conditions": {
        "product_ids": list(intent.product_ids), "unknown_product_ids": list(intent.unknown_product_ids),
        "category": intent.category, "attributes": list(intent.attributes),
        "rating_min": intent.rating_min, "positive_discount": intent.positive_discount,
        "aggregation": intent.aggregation}}

    if intent.strategy == "unsupported":
        diagnostics["reason"] = intent.unsupported_reason
        result = result.iloc[0:0]
    elif intent.strategy == "exact_id":
        order = {value: i for i, value in enumerate(intent.product_ids)}
        result = result[result["product_id"].isin(intent.product_ids)]
        result = result.assign(_order=result["product_id"].map(order)).sort_values("_order")
    elif intent.strategy == "structured":
        if intent.category:
            result = result[result["category"] == intent.category]
        for attribute in intent.attributes:
            result = result[_attribute_mask(result, attribute)]
        if intent.rating_min is not None:
            result = result[result["rating_observed"] & result["rating_numeric"].ge(intent.rating_min)]
        if intent.positive_discount:
            result = result[result["discount_numeric"].gt(0)]
        if intent.aggregation == "min_final_price":
            result = result.sort_values(["final_price_numeric", "product_id"])
        elif intent.aggregation == "max_final_price":
            result = result.sort_values(["final_price_numeric", "product_id"], ascending=[False, True])
        else:
            result = result.sort_values(["final_price_numeric", "product_id"])
    else:
        order = {value: i for i, value in enumerate(dict.fromkeys(map(str, semantic_product_ids)))}
        result = result[result["product_id"].isin(order)]
        result = result.assign(_order=result["product_id"].map(order)).sort_values("_order")
        diagnostics["semantic_candidates_supplied"] = len(order)

    result = result.head(k).drop(columns=["_order"], errors="ignore")
    diagnostics["returned"] = len(result)
    return RetrievalResult(question=question, intent=intent, rows=result, diagnostics=diagnostics)


def structured_retrieve(frame: pd.DataFrame, question: str, k: int = 5) -> pd.DataFrame:
    return route_and_retrieve(frame, question, k=k).rows


def numeric_answer(result: RetrievalResult) -> str:
    """Produce auditable answers for exact/numeric cases without an LLM."""
    rows = result.rows
    if rows.empty:
        return "Structured retrieval is not applicable or returned no rows."
    citations = [f"[{r.source_file}:{int(r.row_id)}]" for r in rows.itertuples()]
    if result.intent.aggregation in {"min_final_price", "max_final_price"}:
        row = rows.iloc[0]
        label = "Minimum" if result.intent.aggregation == "min_final_price" else "Maximum"
        return (f"{label} among all matching rows: {row['title']} (product {row['product_id']}), "
                f"{row['final_price_numeric']:.0f} INR, rating {row['rating_numeric']:.1f}, "
                f"discount {row['discount_numeric']:.0f}% {citations[0]}.")
    parts = []
    for row, citation in zip(rows.itertuples(), citations):
        discount = "not provided" if pd.isna(row.discount_numeric) else f"{row.discount_numeric:.0f}%"
        parts.append(f"{row.title} ({row.product_id}): {row.final_price_numeric:.0f} INR; "
                     f"discount {discount}; rating {row.rating_numeric:.1f} {citation}")
    return "\n".join(parts)
