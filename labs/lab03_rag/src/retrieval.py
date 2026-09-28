"""Redis search and transparent result tables. Scores are distances (lower is better)."""
from time import perf_counter
import pandas as pd


def category_filter(category):
    if category is None:
        return None
    from redisvl.query.filter import Tag
    return Tag("category") == category


def make_retriever(store, k=5, category=None):
    if k < 1:
        raise ValueError("k must be positive")
    return store.as_retriever(search_type="similarity", search_kwargs=dict(k=k, filter=category_filter(category)))


def retrieve(store, query, k=5, category=None):
    if k < 1:
        raise ValueError("k must be positive")
    started = perf_counter()
    pairs = store.similarity_search_with_score(query, k=k, filter=category_filter(category))
    elapsed = (perf_counter() - started) * 1000
    return pairs, elapsed


def result_table(query, pairs, elapsed_ms):
    rows = []
    for rank, (doc, distance) in enumerate(pairs, 1):
        rows.append(dict(query=query, rank=rank, cosine_distance=float(distance),
                         source_file=doc.metadata["source_file"], row_id=doc.metadata["row_id"],
                         record_id=doc.metadata["record_id"], product_id=doc.metadata["product_id"],
                         category=doc.metadata.get("category", ""),
                         preview=doc.page_content[:400], metadata=dict(doc.metadata),
                         query_elapsed_ms=elapsed_ms))
    return pd.DataFrame(rows)
