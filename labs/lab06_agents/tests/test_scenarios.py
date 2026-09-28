import json
from pydantic import BaseModel
from labs.lab06_agents.src.config import LAB_ROOT
from labs.lab06_agents.src.runner import LocalTrace, save_json
from labs.lab06_agents.src.evaluation import evidence_checks


def test_scenarios():
    scenarios = json.loads((LAB_ROOT / 'experiments/scenarios.json').read_text())
    by_id = {s['scenario_id']: s for s in scenarios}
    assert len(by_id) == len(scenarios) == 13
    assert {'no_tool','lookup','cross_file','multi_step','memory','isolation','unsupported','comparison'} <= {s['category'] for s in scenarios}
    assert by_id['baseline']['user_input'] == by_id['comparison_tools']['user_input']
    assert by_id['baseline']['with_tools'] is False
    assert by_id['memory_set']['session'] == by_id['memory_recall']['session'] != by_id['memory_isolated']['session']
    assert list(by_id).index('memory_set') < list(by_id).index('memory_recall')
    assert 'llava' not in by_id['memory_set']['user_input'].lower()


def test_trace_and_sdk_serialization(tmp_path):
    class Details(BaseModel):
        tokens: int
    save_json(tmp_path / 'nested.json', {'nested': Details(tokens=7)})
    assert json.loads((tmp_path / 'nested.json').read_text())['nested']['tokens'] == 7
    trace = LocalTrace(tmp_path / 'trace.json')
    trace.add('tool_call', tool_name='calculate', tool_arguments='{"expression":"17-13"}')
    trace.add('error', error_type='ValueError')
    assert len(json.loads(trace.path.read_text())) == 2


def test_evaluation_not_substring_numbers():
    run = {'final_output': '117 lab04_report.md', 'status': 'SUCCESS', 'tools_used': []}
    checks = evidence_checks(run, {'expected_facts': ['17']}, [])
    assert checks['expected_fact_presence'] == [False]
    assert checks['calculator_used'] is False
