import pandas as pd
from labs.lab04_stable_diffusion.src.evaluation import performance_summary, scheduler_experiment_plan


def test_scheduler_plan_does_not_claim_unrun_results():
    plan = scheduler_experiment_plan()
    assert set(plan.status) == {"EXECUTED_BASELINE", "NOT_EXECUTED"}
    assert (plan.status == "NOT_EXECUTED").sum() == 2


def test_performance_summary_labels_single_observations():
    frame = pd.DataFrame({"experiment": ["a", "b", "b"], "case_id": ["1", "2", "3"],
                          "generation_time_seconds": [1.0, 2.0, 4.0]})
    result = performance_summary(frame).set_index("experiment")
    assert result.loc["a", "observations"] == 1
    assert pd.isna(result.loc["a", "std_runtime_s"])
    assert result.loc["b", "mean_runtime_s"] == 3.0
