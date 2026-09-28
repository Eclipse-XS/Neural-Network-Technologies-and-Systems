import ast
import inspect
from labs.lab07_final_project.src.experiments import scenarios,run_suite
from labs.lab07_final_project.src.config import INSTRUCTIONS


def test_scenario_coverage():
    cases=scenarios()
    assert len({c['scenario_id'] for c in cases})==len(cases)
    assert {'no-tool','RAG','semantic','cross-source','multi-step','memory','isolation','unsupported'}<={c['category'] for c in cases}
    assert sum(c['baseline_comparison'] for c in cases)>=4
    assert all(c['user_turns'] for c in cases)


def test_labels_not_sent_to_runtime():
    tree=ast.parse(inspect.getsource(run_suite))
    assert not any(isinstance(n,ast.Constant) and n.value in ('expected_sources','expected_facts') for n in ast.walk(tree))
    assert '17 - 13' not in INSTRUCTIONS and '0.542' not in INSTRUCTIONS
