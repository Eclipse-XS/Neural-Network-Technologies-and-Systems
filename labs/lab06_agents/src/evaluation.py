"""Transparent descriptive checks plus explicit, run-specific human review."""
import hashlib
import json
import re
from .config import LAB_ROOT, OUTPUTS, WORKSPACE, INSTRUCTIONS, Settings
from .tools import build_tools


def load_verified():
    batch = json.loads((OUTPUTS / 'metadata/latest_batch.json').read_text())
    all_runs = [json.loads(line) for line in (OUTPUTS / 'runs/runs.jsonl').read_text(encoding='utf-8').splitlines()]
    runs = [r for r in all_runs if r['run_id'] in batch['run_ids']]
    assert [r['run_id'] for r in runs] == batch['run_ids']
    settings = Settings()
    names = [t.name for t in build_tools()]
    for item in json.loads((WORKSPACE / 'manifest.json').read_text()):
        assert hashlib.sha256((WORKSPACE / item['workspace_filename']).read_bytes()).hexdigest() == item['sha256']
    for r in runs:
        assert r['tools_available'] == (names if r['agent_variant'] == 'with_tools' else [])
        assert (r['temperature'], r['max_turns'], r['max_tokens'], r['base_url']) == (settings.temperature, settings.max_turns, settings.max_tokens, settings.base_url)
        if settings.model_id:
            assert r['model_id'] == settings.model_id
        assert r['instructions_sha256'] == hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()
        assert r['workspace_sha256'] == hashlib.sha256((WORKSPACE / 'manifest.json').read_bytes()).hexdigest()
        assert r['tracing_disabled'] and r['api_mode'] == 'chat_completions'
        trace = json.loads((LAB_ROOT / r['trace_path']).read_text(encoding='utf-8'))
        assert [e['tool_name'] for e in trace if e['event_type'] == 'tool_call'] == r['tools_used']
    assert len({(r['model_id'], r['temperature'], r['max_tokens']) for r in runs}) == 1
    return runs, batch


def evidence_checks(run, scenario, trace):
    """These checks detect evidence; they are not a semantic correctness score."""
    answer = (run['final_output'] or '').casefold()
    facts = scenario['expected_facts']
    present = [bool(re.search(r'(?<!\d)' + re.escape(fact) + r'(?!\d)', answer)) if fact.isdigit() else fact.casefold() in answer for fact in facts]
    calls = [e for e in trace if e['event_type'] == 'tool_call']
    signatures = [(e['tool_name'], e['tool_arguments']) for e in calls]
    return {'expected_fact_presence': present,
            'citations': sorted(set(re.findall(r'lab0[345]_report\.md', answer))),
            'repeated_identical_calls': len(signatures) - len(set(signatures)),
            'calculator_used': 'calculate' in run['tools_used'],
            'tool_failed': run['status'] != 'SUCCESS' or run.get('tool_error_count', 0) > 0,
            'read_or_search_used': any(n in run['tools_used'] for n in ['search_workspace', 'read_workspace_file'])}


def evaluated_rows():
    runs, _ = load_verified()
    scenarios = {s['scenario_id']: s for s in json.loads((LAB_ROOT / 'experiments/scenarios.json').read_text(encoding='utf-8'))}
    reviews = json.loads((LAB_ROOT / 'report/manual_evaluation.json').read_text(encoding='utf-8'))
    result = []
    for r in runs:
        trace = json.loads((LAB_ROOT / r['trace_path']).read_text(encoding='utf-8'))
        review = reviews[r['run_id']]
        result.append({**r, **evidence_checks(r, scenarios[r['scenario_id']], trace), **review})
    return result
