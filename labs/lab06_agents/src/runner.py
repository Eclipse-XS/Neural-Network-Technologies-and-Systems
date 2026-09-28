"""Persist observable SDK events, including partial trajectories on errors."""
import asyncio
import dataclasses
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from agents import Runner, RunConfig, RunHooks
from .config import OUTPUTS, WORKSPACE, INSTRUCTIONS


def json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Unsupported log value: {type(value).__name__}")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


class LocalTrace(RunHooks):
    def __init__(self, path):
        self.path = path
        self.events = []

    def add(self, event_type, **data):
        self.events.append(dict(step_index=len(self.events) + 1, event_type=event_type, **data))
        save_json(self.path, self.events)

    async def on_tool_start(self, context, agent, tool):
        self.add("tool_call", tool_name=tool.name, call_id=context.tool_call_id,
                 tool_arguments=context.tool_arguments)

    async def on_tool_end(self, context, agent, tool, result):
        error_type = None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                error_type = parsed.get('error_type') if isinstance(parsed, dict) else None
            except ValueError:
                pass
        self.add("tool_result", tool_name=tool.name, call_id=context.tool_call_id,
                 tool_output=result, tool_output_preview=str(result)[:450], error_type=error_type)


async def execute(agent, settings, scenario, session=None, batch_id="manual"):
    run_id = uuid.uuid4().hex
    trace = LocalTrace(OUTPUTS / "traces" / f"{run_id}.json")
    trace.add("user_input", text=scenario["user_input"])
    record = dict(run_id=run_id, batch_id=batch_id, scenario_id=scenario["scenario_id"],
        category=scenario["category"], tool_required=scenario["tool_required"],
        agent_variant="with_tools" if agent.tools else "without_tools",
        session_id=session.session_id if session else None, user_input=scenario["user_input"],
        final_output=None, status="FAILED", model_id=settings.model_id,
        temperature=settings.temperature, max_turns=settings.max_turns, max_tokens=settings.max_tokens,
        base_url=settings.base_url, api_mode="chat_completions", tracing_disabled=True,
        instructions_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        workspace_sha256=hashlib.sha256((WORKSPACE / "manifest.json").read_bytes()).hexdigest(),
        tools_available=[t.name for t in agent.tools], error_type=None, usage=None,
        created_at=datetime.now(timezone.utc).isoformat())
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(Runner.run(agent, scenario["user_input"], session=session,
            max_turns=settings.max_turns, hooks=trace,
            run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)),
            settings.run_timeout_seconds)
        record["final_output"] = result.final_output
        record["status"] = "SUCCESS"
        record["usage"] = dataclasses.asdict(result.context_wrapper.usage)
        record["sdk_item_types"] = [i.type for i in result.new_items]
        trace.add("final_output", text=result.final_output)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        trace.add("error", error_type=type(exc).__name__, message=str(exc))
    record["runtime_seconds"] = round(time.perf_counter() - start, 3)
    calls = [e for e in trace.events if e["event_type"] == "tool_call"]
    record["tools_used"] = [e["tool_name"] for e in calls]
    record["tool_call_count"] = len(calls)
    record["tool_error_count"] = sum(bool(e.get('error_type')) for e in trace.events if e['event_type'] == 'tool_result')
    record["trace_path"] = f"outputs/traces/{run_id}.json"
    path = OUTPUTS / "runs/runs.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
    print(f"{scenario['scenario_id']}: {record['status']} {record['runtime_seconds']}s {record['tools_used']}", flush=True)
    return record
