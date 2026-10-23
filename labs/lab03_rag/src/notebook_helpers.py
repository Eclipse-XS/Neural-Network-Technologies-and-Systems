"""Executable notebook computations shared by full and portfolio-replay modes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import evaluate_id_lists
from .documents import dataframe_to_documents
from .evaluation import verify_answer
from .hybrid import route_and_retrieve


STATUS_LIVE = "LIVE COMPUTATION"
STATUS_REPLAY = "RECOMPUTED FROM EVIDENCE"
STATUS_HISTORICAL = "HISTORICAL INFERENCE"


def computation_status(mode: str, *, historical: bool = False) -> str:
    if historical:
        return STATUS_HISTORICAL
    return STATUS_LIVE if mode == "full" else STATUS_REPLAY


def eda_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute all descriptive tables from the current catalog object."""
    missing = (frame.replace("", np.nan).isna().sum().sort_values(ascending=False)
               .rename("missing").to_frame())
    missing["share"] = missing["missing"] / len(frame)
    duplicates = pd.DataFrame({
        "exact_rows": [int(frame.drop(columns=["source_file", "row_id"], errors="ignore").duplicated().sum())],
        "product_ids": [int(frame["product_id"].duplicated().sum())],
    })
    categories = (frame.groupby("category", as_index=False)
                  .agg(products=("product_id", "size"),
                       median_price_inr=("final_price_numeric", "median"),
                       observed_ratings=("rating_observed", "sum"),
                       median_description_chars=("description_chars", "median")))
    prices = frame["final_price_numeric"].describe(percentiles=[.25, .5, .75, .95]).to_frame().T
    return {"missing": missing, "duplicates": duplicates, "categories": categories, "prices": prices}


def plot_catalog_eda(frame: pd.DataFrame):
    """Build an EDA dashboard from the in-memory DataFrame, never from a stored PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    frame["category"].value_counts().plot.bar(ax=axes[0, 0], color="#3568A8")
    axes[0, 0].set(title="Products by category", ylabel="rows")
    axes[0, 1].hist(frame["final_price_numeric"].dropna(), bins=24, color="#2A9D8F")
    axes[0, 1].set(title="Final-price distribution", xlabel="INR", ylabel="rows")
    observed = frame.loc[frame["rating_observed"], "rating_numeric"].dropna()
    axes[1, 0].hist(observed, bins=np.arange(0, 5.25, .25), color="#E9C46A")
    axes[1, 0].set(title="Observed ratings", xlabel="stars", ylabel="rows")
    frame.boxplot(column="description_chars", by="category", ax=axes[1, 1])
    axes[1, 1].set(title="Description length", xlabel="category", ylabel="characters")
    fig.suptitle("Catalog EDA recomputed from the current DataFrame", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pca_coordinates(coordinates: pd.DataFrame, provenance: str):
    required = {"pc1", "pc2", "category", "product_id"}
    if not required <= set(coordinates):
        raise ValueError(f"PCA table lacks {sorted(required - set(coordinates))}")
    fig, ax = plt.subplots(figsize=(8, 6))
    for category, group in coordinates.groupby("category"):
        ax.scatter(group.pc1, group.pc2, label=category, alpha=.75, s=30)
    ax.set(title=f"PCA projection ({provenance})", xlabel="principal component 1",
           ylabel="principal component 2")
    ax.legend(title="category")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def replay_rankings(results: pd.DataFrame) -> dict[str, list[str]]:
    required = {"query_id", "rank", "product_id"}
    if not required <= set(results):
        raise ValueError(f"Retrieval evidence lacks {sorted(required - set(results))}")
    ordered = results.sort_values(["query_id", "rank"])
    return ordered.groupby("query_id")["product_id"].apply(lambda s: [str(x) for x in s]).to_dict()


def evaluate_semantic_and_hybrid(catalog: pd.DataFrame, queries: list[dict],
                                 semantic_results: pd.DataFrame, k: int = 5):
    """Run inference without labels, then cross the evaluation boundary once."""
    semantic_ids = replay_rankings(semantic_results)
    hybrid_ids, routes, rows = {}, [], []
    for query in queries:
        result = route_and_retrieve(catalog, query["question"], semantic_ids.get(query["id"], ()), k)
        ids = result.product_ids
        if result.intent.strategy not in {"unsupported", "semantic"}:
            ids += [x for x in semantic_ids.get(query["id"], ()) if x not in ids]
        hybrid_ids[query["id"]] = ids[:k]
        routes.append({"query_id": query["id"], "strategy": result.intent.strategy,
                       "returned_ids": result.product_ids,
                       "conditions": json.dumps(result.diagnostics["conditions"], ensure_ascii=False)})
        rows.extend({"query_id": query["id"], **row} for row in result.rows.to_dict("records"))
    metrics = pd.concat([
        evaluate_id_lists(queries, semantic_ids, k).assign(configuration="semantic_baseline"),
        evaluate_id_lists(queries, hybrid_ids, k).assign(configuration="hybrid_router"),
    ], ignore_index=True)
    return metrics, pd.DataFrame(routes), pd.DataFrame(rows), hybrid_ids


def answer_audit(answers: pd.DataFrame, queries: list[dict], catalog: pd.DataFrame,
                 ranked_ids: dict[str, list[str]]) -> pd.DataFrame:
    """Recompute lexical field and citation checks over historical answer text."""
    query_map = {q["id"]: q for q in queries}
    rows = []
    for record in answers.to_dict("records"):
        query = query_map[record["query_id"]]
        selected = catalog[catalog.product_id.astype(str).isin(ranked_ids.get(record["query_id"], []))]
        docs = dataframe_to_documents(selected)
        audit = verify_answer(str(record.get("answer", "")), docs, query.get("expected_fields", []))
        checks = audit["field_checks"]
        rows.append({"query_id": record["query_id"], "status": STATUS_HISTORICAL,
                     "retrieved_records": len(docs), "unknown_citations": len(audit["unknown_citations"]),
                     "valid_citation": audit["valid_citation_present"],
                     "fields_present": sum(c["value_present"] is True for c in checks),
                     "fields_expected": len(checks),
                     "expected_citations": sum(c["expected_citation_present"] for c in checks)})
    return pd.DataFrame(rows)


def plot_retrieval_dashboard(metrics: pd.DataFrame):
    summary = (metrics.groupby("configuration", as_index=False)
               .agg(hit_at_5=("hit_at_k", "mean"), mrr=("mrr", "mean"),
                    recall_at_5=("recall_at_k", "mean")))
    ax = summary.set_index("configuration").plot.bar(figsize=(8, 4), ylim=(0, 1.05), rot=0)
    ax.set(title="Metrics recomputed from ranked IDs", ylabel="score")
    ax.grid(axis="y", alpha=.2)
    ax.figure.tight_layout()
    return ax.figure
