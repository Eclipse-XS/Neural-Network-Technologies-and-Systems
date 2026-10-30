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
from .tools import build_tools, safe_calculate


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
    abstained = bool(re.search(r"\b(?:cannot determine|insufficient evidence|not available)\b|не можу визначити|недостатньо (?:даних|доказів)|не містить .*?(?:інформац|даних)|даних .*? не виявлено|не було зазначено", answer, re.I))
    return dict(citations=cited, invalid_citations=invalid, supplied_citations=sorted(supplied),
        citation='FAIL' if invalid and not any(c in supplied for c in cited) else 'PARTIAL' if invalid else
        'PASS' if cited else 'PASS' if abstained else 'FAIL' if supplied else 'NOT_APPLICABLE')


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
        history = await session.get_items() if session else []
        has_explicit_subject = bool(re.search(r"\b(?:lab(?:oratory)?\s*\d+|report|model|system)\b|"
                                              r"\b(?:лабораторн\w*\s*(?:робот\w*)?\s*№?\s*\d+|звіт|модел|систем)",
                                              user_input, re.I))
        ambiguous = not history and not has_explicit_subject and bool(re.search(
            r"\b(?:it|its|that model|this model)\b|\b(?:вона|воно|він|цей|ця)\b", user_input, re.I))
        if ambiguous:
            answer = "Please clarify which model, system, or laboratory report you mean. A new session has no antecedent for that reference."
            if session:
                await session.add_items([{'role':'user','content':user_input},{'role':'assistant','content':answer}])
            record.update(final_answer=answer, status='SUCCESS', usage=None)
            trace.add('ambiguity_guard', reason='pronoun_without_session_antecedent')
        else:
            result = await asyncio.wait_for(Runner.run(agent, user_input, session=session, max_turns=settings.max_turns,
                hooks=trace, run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)), settings.run_timeout_seconds)
            answer = str(result.final_output)
            record.update(final_answer=answer, status='SUCCESS',
                usage=json.loads(json.dumps(result.context_wrapper.usage, default=serializable)))
            arithmetic_intent = bool(re.search(
                r"\b(?:calculate|difference|sum|percentage|percent|larger|smaller|by how many|total of)\b|"
                r"різниц|сум|відсот|на скільки|більш[аеи]|менш[аеи]", user_input, re.I))
            used_tools = [e['tool_name'] for e in trace.events if e['event_type']=='tool_call']
            if arithmetic_intent and 'calculate' not in used_tools:
                evidence = [c for e in trace.events if e['event_type']=='tool_result' and e['tool_name']=='search_knowledge'
                            and isinstance(e['result'],dict) for c in e['result'].get('chunks',[])]
                if evidence:
                    recovery_prompt = json.dumps({
                        'original_question': user_input,
                        'draft_answer': answer,
                        'retrieved_evidence': evidence,
                        'required_action': 'Verify operands from evidence, call calculate for every derived value, then answer with exact supplied citations. If operands are not supported, abstain.'
                    }, ensure_ascii=False)
                    recovered = await asyncio.wait_for(Runner.run(agent,recovery_prompt,max_turns=4,hooks=trace,
                        run_config=RunConfig(tracing_disabled=True,trace_include_sensitive_data=False)),settings.run_timeout_seconds)
                    answer = str(recovered.final_output)
                    record['final_answer'] = answer
                    trace.add('arithmetic_recovery', reason='derived_arithmetic_without_calculate', corrected_answer=answer)
                    used_tools = [e['tool_name'] for e in trace.events if e['event_type']=='tool_call']
                    if 'calculate' not in used_tools:
                        by_source = {}
                        answer_numbers = {float(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?', answer)}
                        for chunk in evidence:
                            source = chunk['source_id']
                            direct = {float(m) for m in re.findall(r'(\d+)\s*(?:унікальн\w*|unique|разом|total)', chunk['text'], re.I)}
                            if not direct:
                                direct = {float(m) for m in re.findall(r'(?:унікальн\w*|unique|разом|total)\s*(?::\s*)?(\d+)', chunk['text'], re.I)}
                            if direct and (direct & answer_numbers):
                                by_source.setdefault(source, set()).update(direct & answer_numbers)
                            else:
                                relevant = ' '.join(part for part in re.split(r'(?<=[.!?])\s+|\n',chunk['text'])
                                                    if re.search(r'unique|унікальн|загальн.*кільк|total',part,re.I))
                                values = {float(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?', relevant)}
                                values.discard(float(chunk['lab_number']))
                                by_source.setdefault(source,set()).update(values & answer_numbers)
                        operands = [next(iter(values)) for source,values in sorted(by_source.items()) if len(values)==1]
                        if len(operands)==2:
                            if re.search(r'difference|by how many|різниц|на скільки', user_input, re.I):
                                expression = f'{max(operands):g} - {min(operands):g}'
                            elif re.search(r'\bsum\b|сум', user_input, re.I):
                                expression = f'{operands[0]:g} + {operands[1]:g}'
                            else:
                                expression = None
                            if expression:
                                call_id = 'policy_' + uuid.uuid4().hex[:12]
                                trace.add('tool_call',tool_name='calculate',call_id=call_id,
                                          arguments=json.dumps({'expression':expression}),policy_enforced=True)
                                calculation = safe_calculate(expression)
                                trace.add('tool_result',tool_name='calculate',call_id=call_id,result=calculation,
                                          policy_enforced=True)
                                verifier = Agent(name='Arithmetic result editor',model=agent.model,tools=[],instructions="""Answer the original question using only retrieved evidence and the verified calculator result. Cite only exact supplied citations. Never create identifiers or recalculate in prose.""")
                                final_prompt=json.dumps({'question':user_input,'evidence':evidence,'calculation':calculation},ensure_ascii=False)
                                final_result=await asyncio.wait_for(Runner.run(verifier,final_prompt,max_turns=2,hooks=trace,
                                    run_config=RunConfig(tracing_disabled=True,trace_include_sensitive_data=False)),settings.run_timeout_seconds)
                                answer=str(final_result.final_output); record['final_answer']=answer
                                trace.add('arithmetic_policy_result',corrected_answer=answer)
            audit = validate_citations(answer, trace.events)
            evidence = [c for e in trace.events if e['event_type']=='tool_result' and e['tool_name']=='search_knowledge'
                        and isinstance(e['result'],dict) for c in e['result'].get('chunks',[])]
            # Citation syntax alone cannot detect a false claim attached to a valid chunk.
            # Every evidence-backed answer therefore gets a claim/evidence editing pass.
            if evidence and not ('calculate' in used_tools and audit['citation'] == 'PASS'):
                verifier = Agent(name='Grounded answer editor', model=agent.model, tools=[], instructions="""Rewrite the draft using only the supplied evidence. Remove unsupported claims. Cite factual claims only with exact citation strings shown with the evidence. Never output placeholders, filenames, nested brackets, or new identifiers. Prefer explicit observed behavior over generic limitations. Do not describe deliberate tool permission boundaries as model weaknesses. Distinguish unique executions from table rows and reused results. If evidence is insufficient, abstain clearly. Return only the corrected answer.""")
                prompt = json.dumps({'question':user_input,'draft':answer,'evidence':evidence},ensure_ascii=False)
                corrected = await asyncio.wait_for(Runner.run(verifier,prompt,max_turns=2,hooks=trace,
                    run_config=RunConfig(tracing_disabled=True,trace_include_sensitive_data=False)),settings.run_timeout_seconds)
                record['final_answer'] = str(corrected.final_output)
                trace.add('grounding_review', trigger=audit['citation'], original_answer=answer,
                          corrected_answer=record['final_answer'])
                revised_audit = validate_citations(record['final_answer'], trace.events)
                if revised_audit['citation'] in {'FAIL', 'PARTIAL'}:
                    retry_prompt = json.dumps({'question':user_input,'draft':record['final_answer'],
                        'evidence':evidence,'required':'Return a concise supported answer with exact supplied citation strings.'},
                        ensure_ascii=False)
                    corrected = await asyncio.wait_for(Runner.run(verifier,retry_prompt,max_turns=2,hooks=trace,
                        run_config=RunConfig(tracing_disabled=True,trace_include_sensitive_data=False)),settings.run_timeout_seconds)
                    record['final_answer'] = str(corrected.final_output)
                    trace.add('citation_retry', corrected_answer=record['final_answer'])
            # Canonicalize formatting only; this never creates a citation not supplied by retrieval.
            record['final_answer'] = record['final_answer'].replace('[source_id#chunk_id]', '')
            record['final_answer'] = re.sub(
                r'\[(?:Джерело|Citation|Source):\s*`?(\[lab\d+#chunk-[0-9a-f]+\])`?\]', r'\1', record['final_answer'], flags=re.I)
            record['final_answer'] = re.sub(
                r'(\[lab\d+#chunk-[0-9a-f]+\])\(\1\)', r'\1', record['final_answer'])
            record['final_answer'] = re.sub(
                r'(\[lab\d+#chunk-[0-9a-f]+\])\(lab\d+#chunk-[0-9a-f]+\)', r'\1', record['final_answer'])
            record['final_answer'] = re.sub(
                r'\[calculation:\s*([^\]]+)\]\(\[calculation:[^\]]+\]\)', r'\1', record['final_answer'], flags=re.I)
            record['final_answer'] = re.sub(r'\*\*\s*\*\*', '', record['final_answer'])
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
