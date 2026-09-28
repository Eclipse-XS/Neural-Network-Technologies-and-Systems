import asyncio
import json
from types import SimpleNamespace
from agents import MaxTurnsExceeded
from labs.lab06_agents.src import runner
from labs.lab06_agents.src.config import Settings


def test_max_turn_failure_persisted(monkeypatch, tmp_path):
    async def fail(*args, **kwargs):
        raise MaxTurnsExceeded('Test turn limit')
    monkeypatch.setattr(runner.Runner, 'run', fail)
    monkeypatch.setattr(runner, 'OUTPUTS', tmp_path)
    scenario = dict(scenario_id='test', category='test', tool_required=True, user_input='Test')
    result = asyncio.run(runner.execute(SimpleNamespace(tools=[]), Settings(model_id='offline-test'), scenario))
    assert result['status'] == 'FAILED'
    assert result['error_type'] == 'MaxTurnsExceeded'
    saved = json.loads((tmp_path / 'runs/runs.jsonl').read_text())
    assert saved['run_id'] == result['run_id']
    trace = json.loads((tmp_path / 'traces' / (result['run_id'] + '.json')).read_text())
    assert [e['event_type'] for e in trace] == ['user_input', 'error']


def test_recoverable_error_trace(tmp_path):
    trace = runner.LocalTrace(tmp_path / 'trace.json')
    context = SimpleNamespace(tool_call_id='1')
    tool = SimpleNamespace(name='read_workspace_file')
    asyncio.run(trace.on_tool_end(context, None, tool, '{"error_type":"ValueError","error":"not allowed"}'))
    assert trace.events[0]['error_type'] == 'ValueError'
