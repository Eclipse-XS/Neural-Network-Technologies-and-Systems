import json
import pandas as pd

from labs.lab03_rag.src.analysis import evaluate_id_lists, structured_retrieve


def test_evaluate_id_lists_exposes_coverage_ceiling():
    queries = [{"id": "q", "kind": "semantic", "relevant_product_ids": [str(i) for i in range(9)]}]
    result = evaluate_id_lists(queries, {"q": [str(i) for i in range(5)]}, k=5).iloc[0]
    assert result.hit_at_k == 1
    assert result.recall_at_k == 5 / 9
    assert result.recall_at_k < 1


def test_structured_minimum_and_exact_ids():
    frame = pd.DataFrame([
        {"product_id": "a", "category": "sports-shoes", "rating_numeric": 4.1,
         "discount_numeric": 20.0, "final_price_numeric": 900.0},
        {"product_id": "b", "category": "sports-shoes", "rating_numeric": 4.2,
         "discount_numeric": 30.0, "final_price_numeric": 659.0},
    ])
    query = {"id": "q05", "kind": "multi_condition", "relevant_product_ids": ["b"]}
    assert structured_retrieve(frame, query).iloc[0].product_id == "b"
