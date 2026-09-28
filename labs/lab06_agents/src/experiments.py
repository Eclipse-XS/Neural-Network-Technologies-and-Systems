"""Run fixed scenarios serially; fresh sessions for every memory experiment batch."""
import asyncio
import argparse
import json
import uuid
from agents import SQLiteSession
from .agent import connect, create_coursework_agent
from .config import LAB_ROOT, OUTPUTS, Settings
from .runner import execute, save_json


async def run_suite():
    settings = Settings()
    try:
        client, settings = await connect(settings)
    except Exception as exc:
        save_json(OUTPUTS / 'metadata/startup_error.json', {
            'status': 'FAILED', 'error_type': type(exc).__name__, 'message': str(exc)})
        raise
    batch = uuid.uuid4().hex
    settings.session_db.parent.mkdir(parents=True, exist_ok=True)
    sessions = {key: SQLiteSession(f'{batch}_{key}', settings.session_db) for key in ['A', 'B']}
    scenarios = json.loads((LAB_ROOT / 'experiments/scenarios.json').read_text(encoding='utf-8'))
    records = []
    try:
        for scenario in scenarios:
            agent = create_coursework_agent(client, settings, scenario.get('with_tools', True))
            session = sessions.get(scenario.get('session'))
            records.append(await execute(agent, settings, scenario, session, batch))
        memory = {key: {'session_id': session.session_id, 'items': await session.get_items()} for key, session in sessions.items()}
        save_json(OUTPUTS / f'metadata/memory_{batch}.json', memory)
        save_json(OUTPUTS / 'metadata/latest_batch.json', {'batch_id': batch, 'run_ids': [r['run_id'] for r in records], 'memory_path': f'outputs/metadata/memory_{batch}.json'})
    finally:
        await client.close()
        for session in sessions.values():
            session.close()
    return records


async def append_scenario(scenario_id):
    """Add one explicitly selected independent scenario to the verified current batch."""
    from .evaluation import load_verified
    runs, batch = load_verified()
    scenarios = json.loads((LAB_ROOT / 'experiments/scenarios.json').read_text(encoding='utf-8'))
    scenario = next(s for s in scenarios if s['scenario_id'] == scenario_id)
    if scenario.get('session') or any(r['scenario_id'] == scenario_id for r in runs):
        raise ValueError('Only a new independent scenario may be appended')
    client, settings = await connect(Settings())
    try:
        assert (settings.model_id, settings.temperature, settings.max_tokens) == (runs[0]['model_id'], runs[0]['temperature'], runs[0]['max_tokens'])
        record = await execute(create_coursework_agent(client, settings, scenario.get('with_tools', True)), settings, scenario, batch_id=batch['batch_id'])
        batch['run_ids'].append(record['run_id'])
        save_json(OUTPUTS / 'metadata/latest_batch.json', batch)
        return record
    finally:
        await client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--append-scenario')
    args = parser.parse_args()
    asyncio.run(append_scenario(args.append_scenario) if args.append_scenario else run_suite())
