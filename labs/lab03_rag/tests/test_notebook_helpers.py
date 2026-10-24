import json
from pathlib import Path

import pandas as pd

from labs.lab03_rag.src.analysis import load_catalog
from labs.lab03_rag.src.notebook_helpers import eda_tables, evaluate_semantic_and_hybrid


ROOT = Path(__file__).resolve().parents[1]


def test_live_helpers_compute_metrics_without_aggregate_input():
    catalog = load_catalog(ROOT / 'data/raw/csv')
    queries = json.loads((ROOT / 'queries.json').read_text(encoding='utf-8'))['queries']
    evidence = pd.read_csv(ROOT / 'artifacts/portfolio/semantic_results.csv', dtype={'product_id': str})
    metrics, routes, rows, _ = evaluate_semantic_and_hybrid(catalog, queries, evidence)
    assert len(metrics) == 16
    assert set(metrics.configuration) == {'semantic_baseline', 'hybrid_router'}
    assert len(routes) == 8
    assert not rows.empty
    tables = eda_tables(catalog)
    assert tables['duplicates'].iloc[0].product_ids == 0
    assert tables['categories'].products.sum() == len(catalog)
