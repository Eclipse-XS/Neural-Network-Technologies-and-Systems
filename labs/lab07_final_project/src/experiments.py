"""Fixed retrieval/agent experiments; labels are never passed to the agent."""
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import uuid
import pandas as pd
from .config import LAB_ROOT, OUTPUTS, INSTRUCTIONS, Settings
from .knowledge_base import KnowledgeBase, save_json
from .agent import connect, create_agent, create_session, execute


def scenarios():
    return json.loads((LAB_ROOT / 'experiments/scenarios.json').read_text(encoding='utf-8'))


def experiment_fingerprint(settings, kb):
    inputs = dict(settings=dataclasses.asdict(settings), corpus=kb.info['corpus_fingerprint'], instructions=INSTRUCTIONS,
        scenarios=scenarios(), source={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [LAB_ROOT/'src/agent.py',LAB_ROOT/'src/tools.py',LAB_ROOT/'src/knowledge_base.py',LAB_ROOT/'src/experiments.py']})
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def environment(settings, kb):
    import torch
    packages = {}
    for name in ['torch','transformers','openai','openai-agents','langchain','langchain-core','langchain-community',
        'langchain-huggingface','langchain-redis','langchain-text-splitters','sentence-transformers','redis','redisvl','httpx2']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages,
        cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        docker=subprocess.check_output(['docker','version','--format','{{.Server.Version}}'],text=True).strip(),
        compose=subprocess.check_output(['docker','compose','version'],text=True).strip(),
        redis_server=kb.info['redis_version'], lm_studio_url=settings.base_url, model_id=settings.model_id,
        embedding_model=settings.embedding_model, embedding_device='cpu', cloud_tracing=False)


def retrieval_experiment(kb):
    queries = json.loads((LAB_ROOT / 'experiments/retrieval_queries.json').read_text(encoding='utf-8'))
    rows, summary = [], []
    for q in queries:
        start = time.perf_counter()
        chunks = kb.search(q['query'])
        elapsed = (time.perf_counter() - start) * 1000
        found = {c.source_id for c in chunks}
        expected = set(q['expected_source_ids'])
        summary.append(dict(query_id=q['query_id'], hit_at_k=bool(found & expected) if expected else None,
            all_sources_hit=expected <= found if expected else None, expected=sorted(expected), retrieved=sorted(found),
            first_expected_rank=next((i for i,c in enumerate(chunks,1) if c.source_id in expected),None),runtime_ms=elapsed))
        for rank,c in enumerate(chunks,1):
            rows.append(dict(query_id=q['query_id'],query=q['query'],rank=rank,source=c.source_id,section=c.section,
                chunk_id=c.chunk_id,score_or_distance=c.cosine_distance,preview=c.text[:300], text=c.text, runtime_ms=elapsed))
    pd.DataFrame(rows).to_csv(OUTPUTS / 'retrieval/retrieval_results.csv',index=False)
    eligible = [x for x in summary if x['hit_at_k'] is not None]
    result = dict(query_count=len(queries),k=kb.settings.top_k,eligible=len(eligible),
        source_hits=sum(x['hit_at_k'] for x in eligible),all_source_hits=sum(x['all_sources_hit'] for x in eligible),
        source_hit_at_k=sum(x['hit_at_k'] for x in eligible)/len(eligible),queries=summary,
        interpretation='Source-level diagnostic only; does not test answer-bearing chunk coverage.')
    save_json(OUTPUTS / 'retrieval/metrics.json',result)
    return result


async def preflight(client, settings, kb):
    ordinary = await client.chat.completions.create(model=settings.model_id,
        messages=[dict(role='user',content='Reply with READY only.')],temperature=settings.temperature,max_tokens=32)
    save_json(OUTPUTS / 'metadata/chat_preflight.json',dict(model=settings.model_id,answer=ordinary.choices[0].message.content))
    agent = create_agent(client,settings,kb)
    result = await execute(agent,settings,'Calculate (21 + 7) / 4 using your calculator.',scenario_id='preflight_calculator',batch_id='preflight')
    save_json(OUTPUTS / 'metadata/tool_preflight.json',result)
    if result['status'] != 'SUCCESS' or 'calculate' not in result['tools_used']:
        raise RuntimeError('Native function calling preflight failed; inspect persisted trace')
    return result


async def run_suite(settings: Settings, kb: KnowledgeBase, *, force=False):
    client, settings = await connect(settings)
    try:
        fingerprint = experiment_fingerprint(settings,kb)
        cache = OUTPUTS / 'runs/batch.json'
        if cache.exists() and not force:
            batch = json.loads(cache.read_text(encoding='utf-8'))
            if batch['fingerprint'] == fingerprint:
                records = load_batch(batch)
                print('Reusing verified real experiment batch',batch['batch_id'],flush=True)
                return records
        save_json(OUTPUTS / 'metadata/environment.json',environment(settings,kb))
        await preflight(client,settings,kb)
        agent, plain = create_agent(client,settings,kb), create_agent(client,settings)
        batch_id = 'experiment_' + uuid.uuid4().hex[:12]
        records = []
        for scenario in scenarios():
            session = create_session(settings,batch_id + '_' + scenario['scenario_id'])
            for turn_index,question in enumerate(scenario['user_turns']):
                # Deliberately pass only the user text, never expected facts/sources.
                result = await execute(agent,settings,question,scenario_id=scenario['scenario_id'],
                    session=session,batch_id=batch_id,fingerprint=fingerprint)
                records.append(result)
                print(scenario['scenario_id'],turn_index,result['status'],result['tools_used'],result['runtime_seconds'],flush=True)
            if scenario['requires_memory'] or scenario['category'] == 'isolation':
                save_json(OUTPUTS / f'metadata/memory_{scenario["scenario_id"]}_{batch_id}.json',await session.get_items())
            if scenario['baseline_comparison']:
                result = await execute(plain,settings,scenario['user_turns'][0],scenario_id=scenario['scenario_id'],
                    session=create_session(settings,batch_id + '_plain_' + scenario['scenario_id']),
                    batch_id=batch_id,fingerprint=fingerprint)
                records.append(result)
                print(scenario['scenario_id'],'plain',result['status'],result['runtime_seconds'],flush=True)
        batch = dict(batch_id=batch_id,fingerprint=fingerprint,run_ids=[r['run_id'] for r in records],
            trace_hashes={r['run_id']:hashlib.sha256((LAB_ROOT/r['trace_path']).read_bytes()).hexdigest() for r in records},
            record_hashes={r['run_id']:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() for r in records})
        save_json(cache,batch)
        return records
    finally:
        await client.close()


def load_batch(batch=None):
    if batch is None:
        batch=json.loads((OUTPUTS/'runs/batch.json').read_text(encoding='utf-8'))
    records={r['run_id']:r for r in [json.loads(line) for line in (OUTPUTS/'runs/runs.jsonl').read_text(encoding='utf-8').splitlines()]}
    selected=[records[i] for i in batch['run_ids']]
    for r in selected:
        if hashlib.sha256((LAB_ROOT/r['trace_path']).read_bytes()).hexdigest()!=batch['trace_hashes'][r['run_id']]:
            raise ValueError('Cached trace integrity failure')
        if hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()!=batch['record_hashes'][r['run_id']]:
            raise ValueError('Cached run integrity failure')
    return selected


async def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--retrieval-only',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    settings=Settings();kb=KnowledgeBase(settings)
    print(kb.initialize(),flush=True)
    print(retrieval_experiment(kb),flush=True)
    if not args.retrieval_only:
        await run_suite(settings,kb,force=args.force)


if __name__ == '__main__':
    asyncio.run(main())
