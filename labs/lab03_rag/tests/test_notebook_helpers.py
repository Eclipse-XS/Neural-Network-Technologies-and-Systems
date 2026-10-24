import json
from pathlib import Path

import pandas as pd

from labs.lab03_rag.src.analysis import load_catalog
from labs.lab03_rag.src.notebook_helpers import eda_tables, evaluate_semantic_and_hybrid


ROOT = Path(__file__).resolve().parents[1]


def test_live_helpers_compute_metrics_without_aggregate_input(tmp_path):
    # Raw CSVs are intentionally gitignored. Reconstruct the same three-file
    # boundary from the published row-level snapshot so CI tests real ingestion
    # semantics instead of depending on a developer's local dataset directory.
    snapshot = pd.read_csv(ROOT / 'artifacts/portfolio/catalog_replay.csv', dtype=str,
                           keep_default_na=False, encoding='utf-8-sig')
    for source_file, rows in snapshot.groupby('source_file'):
        rows.drop(columns=['source_file', 'row_id']).to_csv(
            tmp_path / source_file, index=False, encoding='utf-8-sig')
    catalog = load_catalog(tmp_path)
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
