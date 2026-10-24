import builtins
import inspect

import pandas as pd

from labs.lab03_rag.src.analysis import evaluate_id_lists
from labs.lab03_rag.src import hybrid
from labs.lab03_rag.src.hybrid import parse_intent, route_and_retrieve


def catalog():
    rows = [
        ("17745816", "Lancer", "sports-shoes", 4.0, 40.0, 1999.0, "running shoes", "Textile"),
        ("11960808", "Campus", "sports-shoes", 4.2, 56.0, 659.0, "walking shoes", "Mesh"),
        ("15720652", "Rubans", "earrings", 3.8, 76.0, 355.0, "silver-plated earrings", "Silver-Plated"),
        ("19072820", "FemNmas", "earrings", 4.1, 50.0, 249.0, "silver plated earrings", "Silver-Plated"),
        ("21242962", "Tommy Hilfiger", "dresses", 4.3, 10.0, 5699.0, "maxi dress", "Cotton"),
    ]
    frame = pd.DataFrame(rows, columns=["product_id", "title", "category", "rating_numeric",
                                        "discount_numeric", "final_price_numeric",
                                        "product_description", "product_specifications"])
    frame["rating_observed"] = True
    frame["source_file"] = frame.category + ".csv"
    frame["row_id"] = range(len(frame))
    return frame


def test_metrics_expose_recall_ceiling_and_wrong_ground_truth():
    queries = [{"id": "q", "kind": "semantic", "relevant_product_ids": [str(i) for i in range(9)]}]
    result = evaluate_id_lists(queries, {"q": [str(i) for i in range(5)]}, k=5).iloc[0]
    assert result.hit_at_k == 1 and result.recall_at_k == 5 / 9
    wrong = [{**queries[0], "relevant_product_ids": ["wrong"]}]
    degraded = evaluate_id_lists(wrong, {"q": [str(i) for i in range(5)]}, k=5).iloc[0]
    assert degraded.hit_at_k == 0 and degraded.mrr == 0 and degraded.recall_at_k == 0


def test_exact_id_parser_single_two_unknown_duplicate_boundary_and_prices():
    ids = catalog().product_id
    assert parse_intent("Product 17745816", ids).product_ids == ("17745816",)
    assert parse_intent("Compare 17745816 and 15720652", ids).product_ids == ("17745816", "15720652")
    assert parse_intent("Compare 17745816, 17745816", ids).product_ids == ("17745816",)
    assert parse_intent("Product 99999999", ids).unknown_product_ids == ("99999999",)
    assert not parse_intent("Reference 1177458169 and price 1999 INR", ids).product_ids
    assert parse_intent("Price 1999 INR", ids).strategy == "semantic"
    assert parse_intent("Find comfortable footwear", ids).strategy == "semantic"


def test_ground_truth_expected_fields_and_query_id_mutation_do_not_change_retrieval():
    frame = catalog()
    base = {"id": "q04", "question": "Compare 17745816 and 15720652",
            "relevant_product_ids": ["17745816", "15720652"], "expected_fields": ["secret"]}
    mutated = {**base, "id": "renamed", "relevant_product_ids": ["wrong"], "expected_fields": []}
    first = route_and_retrieve(frame, base["question"]).product_ids
    second = route_and_retrieve(frame, mutated["question"]).product_ids
    assert first == second == ["17745816", "15720652"]


def test_paraphrases_use_rules_not_case_ids():
    frame = catalog()
    questions = [
        "Find the lowest final price among sports shoes with rating at least 4 and a positive discount.",
        "Which sports shoes are cheapest when rated at least 4 and discount > 0?",
        "Minimum priced athletic shoes with rating not less than 4 and non-zero discount.",
    ]
    assert [route_and_retrieve(frame, q).product_ids[0] for q in questions] == ["11960808"] * 3


def test_new_product_exact_lookup_needs_no_manifest_update():
    frame = catalog()
    new = frame.iloc[[0]].copy()
    new.loc[:, "product_id"] = "76543210"
    new.loc[:, "title"] = "New product"
    expanded = pd.concat([frame, new], ignore_index=True)
    assert route_and_retrieve(expanded, "Inspect product 76543210").product_ids == ["76543210"]


def test_attribute_filter_and_unsupported_intent():
    frame = catalog()
    result = route_and_retrieve(frame, "Show silver plated earrings ordered by price")
    assert result.intent.strategy == "structured"
    assert result.product_ids == ["19072820", "15720652"]
    unsupported = route_and_retrieve(frame, "Can 17745816 be delivered to Kyiv tomorrow; shipping fee?")
    assert unsupported.intent.strategy == "unsupported" and unsupported.rows.empty
    assert "absent" in unsupported.diagnostics["reason"]


def test_inference_source_and_runtime_are_isolated_from_evaluation_artifacts(monkeypatch):
    source = inspect.getsource(hybrid)
    for forbidden in ("relevant_product_ids", "expected_fields", "expected_answer", "queries.json",
                      "holdout_queries.json", "query[\"id\"]"):
        assert forbidden not in source
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("I/O forbidden")))
    assert route_and_retrieve(catalog(), "Compare 17745816 and 15720652").product_ids == ["17745816", "15720652"]
