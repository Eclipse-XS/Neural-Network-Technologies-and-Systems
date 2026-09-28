"""Local SDK agent, SQLite conversation memory and observable durable traces."""
import asyncio
import dataclasses
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
import httpx2
from openai import AsyncOpenAI
from agents import (Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig,
    RunHooks, SQLiteSession, set_tracing_disabled)
from .config import INSTRUCTIONS, OUTPUTS, Settings
from .knowledge_base import save_json
from .tools import build_tools


async def connect(settings: Settings):
    """Discover actual model IDs with no proxy, redirects or cloud fallback."""
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=settings.base_url, api_key='lm-studio', timeout=settings.timeout_seconds,
        max_retries=0, http_client=httpx2.AsyncClient(trust_env=False, follow_redirects=False))
    try:
        candidates = [m.id for m in (await client.models.list()).data if 'embedding' not in m.id.lower()]
        model_id = settings.model_id or next((m for m in candidates if m == 'mistralai/ministral-3-3b'), candidates[0] if candidates else '')
        if not model_id or model_id not in candidates:
            raise RuntimeError(f'No requested local chat model available: {candidates}')
        return client, dataclasses.replace(settings, model_id=model_id)
    except BaseException:
        await client.close()
        raise


def create_agent(client, settings: Settings, kb=None):
    return Agent(name='Coursework Knowledge Assistant', instructions=INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(settings.model_id, client), tools=build_tools(kb) if kb else [],
        model_settings=ModelSettings(temperature=settings.temperature, max_tokens=settings.max_tokens,
            tool_choice='auto' if kb else None, parallel_tool_calls=False if kb else None))


def create_session(settings: Settings, session_id: str | None = None) -> SQLiteSession:
    settings.session_db.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(session_id or 'interactive_' + uuid.uuid4().hex[:12], settings.session_db)


def serializable(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(type(value).__name__)


class LocalTrace(RunHooks):
    def __init__(self, path):
        self.path = path
        self.events = []

    def add(self, event_type: str, **data):
        self.events.append(dict(step=len(self.events) + 1, event_type=event_type, **data))
        save_json(self.path, self.events)

    async def on_tool_start(self, context, agent, tool):
        self.add('tool_call', tool_name=tool.name, call_id=context.tool_call_id, arguments=context.tool_arguments)

    async def on_tool_end(self, context, agent, tool, result):
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        self.add('tool_result', tool_name=tool.name, call_id=context.tool_call_id, result=result)


def validate_citations(answer: str, events: list[dict]) -> dict:
    """Check identifiers against this turn's retrieval, not merely the corpus."""
    supplied = {c['citation'] for e in events if e['event_type'] == 'tool_result' and e['tool_name'] == 'search_knowledge'
        and isinstance(e['result'], dict) for c in e['result'].get('chunks', [])}
    cited = re.findall(r'\[lab\d+#[^\]\n]+\]', answer)
    invalid = [c for c in cited if c not in supplied]
    # Filename-style or malformed chunk citations must not silently count as valid.
    suspicious = re.findall(r'\[[^\]\n]*(?:chunk-|_report\.md)[^\]\n]*\]', answer)
    invalid += [c for c in suspicious if c not in supplied and c not in invalid]
    return dict(citations=cited, invalid_citations=invalid, supplied_citations=sorted(supplied),
        citation='FAIL' if invalid and not any(c in supplied for c in cited) else 'PARTIAL' if invalid else
        'PASS' if cited else 'FAIL' if supplied else 'NOT_APPLICABLE')


async def execute(agent, settings: Settings, user_input: str, *, scenario_id='interactive',
                  session=None, batch_id='manual', fingerprint=None) -> dict:
    """Persist each run and tool event immediately; failures retain partial traces."""
    run_id = uuid.uuid4().hex
    trace = LocalTrace(OUTPUTS / 'traces' / f'{run_id}.json')
    trace.add('user_input', text=user_input)
    record = dict(run_id=run_id, scenario_id=scenario_id, batch_id=batch_id, fingerprint=fingerprint,
        session_id=session.session_id if session else None, agent_variant='assistant' if agent.tools else 'plain',
        user_input=user_input, final_answer='', status='FAILED', error=None, usage=None,
        model_id=settings.model_id, temperature=settings.temperature, max_tokens=settings.max_tokens,
        max_turns=settings.max_turns, base_url=settings.base_url, api_mode='chat_completions', tracing_disabled=True,
        instructions_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(), trace_path=f'outputs/traces/{run_id}.json')
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(Runner.run(agent, user_input, session=session, max_turns=settings.max_turns,
            hooks=trace, run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)), settings.run_timeout_seconds)
        record.update(final_answer=str(result.final_output), status='SUCCESS',
            usage=json.loads(json.dumps(result.context_wrapper.usage, default=serializable)))
        trace.add('final_answer', text=record['final_answer'])
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        trace.add('error', message=record['error'])
    record['runtime_seconds'] = round(time.perf_counter() - start, 3)
    record['tools_used'] = [e['tool_name'] for e in trace.events if e['event_type'] == 'tool_call']
    record['tool_call_count'] = len(record['tools_used'])
    record.update(validate_citations(record['final_answer'], trace.events))
    record['sources_retrieved'] = sorted({c['source_id'] for e in trace.events if e['event_type'] == 'tool_result'
        and e['tool_name'] == 'search_knowledge' and isinstance(e['result'], dict) for c in e['result'].get('chunks', [])})
    path = OUTPUTS / 'runs/runs.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record
