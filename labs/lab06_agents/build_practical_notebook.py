"""Build the practical Lab 06 notebook; execution is a separate QA step."""
from pathlib import Path

import nbformat


LAB_ROOT = Path(__file__).resolve().parent


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Лабораторна робота №6 — практичний LLM-агент

**Варіант 3: OpenAI Agents SDK.** Сценарій: локальний асистент аналізує звіти лабораторних робіт 3–5, виконує безпечні обчислення та пам'ятає контекст діалогу.

Навчальний workflow: **Environment → local LLM endpoint → Agent → Tools → Memory → Runner → Tool calls → Multi-step execution → Evaluation**.

LLM-агент відрізняється від звичайного одиничного LLM-запиту наявністю керованого циклу дій: модель може повернути structured function call, Runner виконує дозволену Python-функцію, передає результат назад моделі й повторює цикл до фінальної відповіді. `SQLiteSession` додає історію попередніх реплік. OpenAI Agents SDK надає `Agent`, `Runner`, tools, sessions, hooks і guardrails виконання.
"""),
    md(r"""
## 1. Режими, імпорти та версії

- **LIVE** створює нові записи у `outputs/notebook_live/`; це не історичні evidence.
- **REPLAY** лише читає раніше зафіксовану серію з `outputs/runs/` і `outputs/traces/`.
- Один простий LIVE smoke виконується за замовчуванням, якщо endpoint доступний.
- Повна матриця (baseline/tools, memory, multi-step, instructions) виконується лише з `LAB06_RUN_FULL_LIVE=1`, оскільки вона повільна й недетермінована.
"""),
    code(r"""
import asyncio, hashlib, inspect, json, os, platform, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat, pandas as pd
from IPython.display import Markdown, display
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig, SQLiteSession
import importlib.metadata as metadata

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labs.lab06_agents.src.agent import connect, create_coursework_agent
from labs.lab06_agents.src.config import INSTRUCTIONS, LAB_ROOT, OUTPUTS, Settings, WORKSPACE
from labs.lab06_agents.src.evaluation import load_verified
from labs.lab06_agents.src.runner import LocalTrace, execute, save_json
from labs.lab06_agents.src.tools import WorkspaceTools, build_tools, safe_calculate

RUN_LIVE_SMOKE = os.getenv("LAB06_RUN_LIVE_SMOKE", "1") == "1"
RUN_FULL_LIVE = os.getenv("LAB06_RUN_FULL_LIVE", "0") == "1"
LIVE_ROOT = OUTPUTS / "notebook_live"
LIVE_ROOT.mkdir(parents=True, exist_ok=True)

versions = {name: metadata.version(name) for name in
            ["openai-agents", "openai", "pydantic", "httpx2", "nbformat", "pandas", "matplotlib"]}
print("Python:", platform.python_version())
print("Packages:", versions)
print("Modes:", {"LIVE_SMOKE": RUN_LIVE_SMOKE, "FULL_LIVE_MATRIX": RUN_FULL_LIVE, "REPLAY": True})

def load_replay_bundle():
    # Never claim verification after workspace drift.
    try:
        runs, batch = load_verified()
        verification = "REPLAY_VERIFIED"
    except AssertionError:
        batch = json.loads((OUTPUTS / "metadata/latest_batch.json").read_text(encoding="utf-8"))
        all_runs = [json.loads(line) for line in (OUTPUTS / "runs/runs.jsonl").read_text(encoding="utf-8").splitlines()]
        by_run_id = {row["run_id"]: row for row in all_runs}
        runs = [by_run_id[run_id] for run_id in batch["run_ids"]]
        verification = "REPLAY_UNVERIFIED_WORKSPACE_MISMATCH"
    reviews = json.loads((LAB_ROOT / "report/manual_evaluation.json").read_text(encoding="utf-8"))
    evaluated = []
    for row in runs:
        trace = json.loads((LAB_ROOT / row["trace_path"]).read_text(encoding="utf-8"))
        signatures = [(event.get("tool_name"), event.get("tool_arguments"))
                      for event in trace if event["event_type"] == "tool_call"]
        evaluated.append({**row, **reviews.get(row["run_id"], {}),
                          "repeated_identical_calls": len(signatures) - len(set(signatures)),
                          "provenance": verification})
    return evaluated, batch, verification

replay_rows, replay_batch, replay_verification = load_replay_bundle()
print("Historical evidence:", replay_verification)
"""),
    md(r"""
## 2. Перевірка локального OpenAI-compatible endpoint

Перевірка складається з `/v1/models` і реального Chat Completions запиту. Сам TCP connect не вважається доказом працездатності API. Секрети не виводяться. Якщо будь-який крок падає, LIVE отримує статус `BLOCKED_BY_ENVIRONMENT`, а REPLAY продовжує працювати.
"""),
    code(r"""
settings = Settings()
client = None
live_status = "NOT_RUN"
preflight = {"base_url": settings.base_url, "api_mode": "chat_completions", "started_at": datetime.now(timezone.utc).isoformat()}
started = time.perf_counter()
try:
    client, settings = await connect(settings)
    models = [item.id for item in (await client.models.list()).data]
    direct = await client.chat.completions.create(
        model=settings.model_id,
        messages=[{"role": "user", "content": "Reply with exactly READY."}],
        temperature=0,
        max_tokens=16,
    )
    preflight.update(status="PASS", models=models, selected_model=settings.model_id,
                     response=direct.choices[0].message.content)
    live_status = "PASS"
except Exception as exc:
    preflight.update(status="BLOCKED_BY_ENVIRONMENT", error_type=type(exc).__name__, error=str(exc))
    live_status = "BLOCKED_BY_ENVIRONMENT"
preflight["runtime_seconds"] = round(time.perf_counter() - started, 3)
save_json(LIVE_ROOT / "latest_preflight.json", preflight)
display(preflight)
"""),
    md(r"""
## 3. Створення Agent

`name` ідентифікує агента; `instructions` задають роль та обмеження; `model` прив'язаний до явного локального клієнта; `ModelSettings` фіксує temperature, автоматичний вибір tools, заборону parallel calls і token limit. `max_turns` та вимкнення cloud tracing передаються Runner під час запуску.
"""),
    code(r"""
agent = None
if client is not None:
    local_model = OpenAIChatCompletionsModel(settings.model_id, client)
    agent = Agent(
        name="Coursework Assistant",
        instructions=INSTRUCTIONS,
        model=local_model,
        tools=build_tools(),
        model_settings=ModelSettings(
            temperature=settings.temperature,
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=settings.max_tokens,
        ),
    )
    print({"name": agent.name, "model": settings.model_id,
           "tools": [tool.name for tool in agent.tools],
           "temperature": settings.temperature, "max_turns": settings.max_turns})
else:
    print("Agent creation: BLOCKED_BY_ENVIRONMENT")
"""),
    md(r"""
## 4. Реалізація та прямі unit-проби інструментів

Проєкт має чотири реальні read-only tools. Нижче показано реалізацію файлового store й безпечного AST-калькулятора, схеми зареєстрованих функцій, input validation і прямі тести. Прямі тести доводять коректність Python-функцій, але не автономний вибір інструмента моделлю; такий доказ дає trace Runner у наступному розділі.
"""),
    code(r"""
print(inspect.getsource(WorkspaceTools.search))
print(inspect.getsource(safe_calculate))
registered_tools = build_tools()
for tool in registered_tools:
    display({"name": tool.name, "description": tool.description, "schema": tool.params_json_schema})

store = WorkspaceTools()
direct_tool_checks = {
    "list": store.list_files(),
    "search": store.search("Stable Diffusion", "lab04_report.md"),
    "calculate": safe_calculate("(17 - 13) * 2"),
}
try:
    store.read("../README.md")
except ValueError as exc:
    direct_tool_checks["validation_error"] = str(exc)
display(direct_tool_checks)
"""),
    md(r"""
## 5. Перший реальний Runner.run і autonomous tool calling

Це центральний smoke-сценарій. Prompt не називає функцію. `LocalTrace` записує фактичні tool calls і outputs. Якщо модель не викличе інструмент, notebook покаже нуль викликів — нічого не фабрикується.
"""),
    code(r"""
smoke_record = {"status": "NOT_RUN", "provenance": "LIVE"}
if RUN_LIVE_SMOKE and agent is not None:
    smoke_id = uuid.uuid4().hex
    smoke_trace_path = LIVE_ROOT / "traces" / f"{smoke_id}.json"
    smoke_trace = LocalTrace(smoke_trace_path)
    user_request = "Which Stable Diffusion model was used in Lab 4? Cite the local report filename."
    smoke_trace.add("user_input", text=user_request)
    started = time.perf_counter()
    try:
        # Direct SDK execution required by the laboratory methodology.
        result = await Runner.run(
            agent,
            input=user_request,
            max_turns=settings.max_turns,
            hooks=smoke_trace,
            run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
        )
        smoke_trace.add("final_output", text=result.final_output)
        smoke_record = {
            "run_id": smoke_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance": "LIVE", "status": "PASS", "model_id": settings.model_id,
            "endpoint": {"base_url": settings.base_url, "api_mode": "chat_completions"},
            "prompt": user_request, "agent_name": agent.name, "agent_instructions": INSTRUCTIONS,
            "tools_available": [tool.name for tool in agent.tools],
            "tool_calls": [e for e in smoke_trace.events if e["event_type"] == "tool_call"],
            "tool_call_count": sum(e["event_type"] == "tool_call" for e in smoke_trace.events),
            "sdk_item_types": [item.type for item in result.new_items],
            "final_output": result.final_output,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "trace_path": str(smoke_trace_path.relative_to(LAB_ROOT)).replace("\\", "/"),
        }
    except Exception as exc:
        smoke_record = {"run_id": smoke_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "provenance": "LIVE", "status": "FAIL", "model_id": settings.model_id,
                        "prompt": user_request, "error_type": type(exc).__name__, "error": str(exc),
                        "runtime_seconds": round(time.perf_counter() - started, 3)}
    save_json(LIVE_ROOT / "latest_smoke.json", smoke_record)
elif not RUN_LIVE_SMOKE:
    smoke_record["reason"] = "LAB06_RUN_LIVE_SMOKE=0"
else:
    smoke_record.update(status="BLOCKED_BY_ENVIRONMENT", reason=preflight.get("error"))
display(smoke_record)
"""),
    md(r"""
## 6. Контроль: агент без інструментів і з інструментами

Змінюється лише список `tools`; model, prompt, instructions і generation settings однакові. Повний LIVE-контроль запускається явно. Інакше нижче буде REPLAY фактичної історичної пари з чітким provenance.
"""),
    code(r"""
comparison_prompt = "According to the local reports, how many generated/inference outputs were produced in Lab 4 and Lab 5? Which laboratory produced more, and by how many? Cite the source files."
live_comparison = []
if RUN_FULL_LIVE and client is not None:
    for with_tools in [False, True]:
        variant = create_coursework_agent(client, settings, with_tools=with_tools)
        scenario = {"scenario_id": f"notebook_comparison_{with_tools}", "category": "comparison",
                    "tool_required": True, "user_input": comparison_prompt}
        row = await execute(variant, settings, scenario, batch_id="notebook_live")
        row["provenance"] = "LIVE"
        live_comparison.append(row)
    display(pd.DataFrame(live_comparison)[["agent_variant", "status", "tool_call_count", "tools_used", "runtime_seconds", "final_output"]])
else:
    replay_pair = [r for r in replay_rows if r["scenario_id"] in {"baseline", "comparison_tools"}]
    display(pd.DataFrame(replay_pair)[["provenance", "scenario_id", "agent_variant", "status", "tool_call_count", "tools_used", "runtime_seconds", "final_output"]])
"""),
    md(r"""
## 7. Реальна session memory та ізоляція

Session A отримує alias у першому Runner.run і використовує його в другому. Session B має інший ID і не повинна знати alias. Для незалежних експериментів один session ID не перевикористовується.
"""),
    code(r"""
memory_results = []
if RUN_FULL_LIVE and agent is not None:
    memory_db = LIVE_ROOT / "sessions" / f"memory_{uuid.uuid4().hex}.sqlite"
    memory_db.parent.mkdir(parents=True, exist_ok=True)
    session_a = SQLiteSession(f"A_{uuid.uuid4().hex}", memory_db)
    session_b = SQLiteSession(f"B_{uuid.uuid4().hex}", memory_db)
    try:
        for label, prompt, session in [
            ("A_set", "For this conversation, call Lab 5 'Vision Lab'.", session_a),
            ("A_recall", "Which model was used in Vision Lab?", session_a),
            ("B_isolated", "Which model was used in Vision Lab?", session_b),
        ]:
            started = time.perf_counter()
            result = await Runner.run(agent, input=prompt, session=session,
                                      max_turns=settings.max_turns,
                                      run_config=RunConfig(tracing_disabled=True))
            memory_results.append({"step": label, "session_id": session.session_id,
                                   "prompt": prompt, "final_output": result.final_output,
                                   "runtime_seconds": round(time.perf_counter() - started, 3),
                                   "provenance": "LIVE"})
        assert session_a.session_id != session_b.session_id
        display(pd.DataFrame(memory_results))
        display({"A_items": await session_a.get_items(), "B_items": await session_b.get_items()})
    finally:
        session_a.close(); session_b.close()
else:
    memory_replay = [r for r in replay_rows
                     if r["scenario_id"] in {"memory_set", "memory_recall", "memory_isolated"}]
    display(pd.DataFrame(memory_replay)[["provenance", "scenario_id", "session_id", "status", "tool_call_count", "final_output"]])
"""),
    md(r"""
## 8. Багатокрокове завдання

Очікувана observable-послідовність: user input → модельне рішення → tool call → tool result → наступне рішення → final output. Trace не містить прихованого chain-of-thought. Кілька кроків визнаються лише за фактичними events.
"""),
    code(r"""
multi_prompt = "According to the local reports, how many generated/inference outputs were produced in Lab 4 and Lab 5? Which laboratory produced more, and by how many? Cite the source files."
if RUN_FULL_LIVE and agent is not None:
    multi_scenario = {"scenario_id": "notebook_multi", "category": "multi_step",
                      "tool_required": True, "user_input": multi_prompt}
    multi_record = await execute(agent, settings, multi_scenario, batch_id="notebook_live")
    multi_record["provenance"] = "LIVE"
    multi_trace = json.loads((LAB_ROOT / multi_record["trace_path"]).read_text(encoding="utf-8"))
else:
    multi_record = next(r for r in replay_rows if r["scenario_id"] == "multi")
    multi_trace = json.loads((LAB_ROOT / multi_record["trace_path"]).read_text(encoding="utf-8"))
display(multi_record)
display(pd.DataFrame(multi_trace))
print("Actual tool calls:", sum(e["event_type"] == "tool_call" for e in multi_trace))
"""),
    md(r"""
## 9. Вплив system instructions

Контроль змінює тільки instructions. Model, prompt, tools і ModelSettings залишаються однаковими. Строга версія вимагає джерело й забороняє відповідь без evidence; нейтральна — лише визначає роль.
"""),
    code(r"""
instruction_results = []
if RUN_FULL_LIVE and client is not None:
    variants = {
        "neutral": "You are a coursework assistant. Answer the user's question.",
        "strict": INSTRUCTIONS,
    }
    for label, instructions in variants.items():
        variant = Agent(name=f"Instructions {label}", instructions=instructions,
                        model=OpenAIChatCompletionsModel(settings.model_id, client),
                        tools=build_tools(),
                        model_settings=ModelSettings(temperature=settings.temperature, tool_choice="auto",
                                                     parallel_tool_calls=False, max_tokens=settings.max_tokens))
        scenario = {"scenario_id": f"instructions_{label}", "category": "instructions",
                    "tool_required": True, "user_input": "Which multimodal model was used in Lab 5? Cite the local report."}
        row = await execute(variant, settings, scenario, batch_id="notebook_live")
        row["instruction_variant"] = label; row["provenance"] = "LIVE"
        instruction_results.append(row)
    display(pd.DataFrame(instruction_results)[["instruction_variant", "tool_call_count", "tools_used", "status", "runtime_seconds", "final_output"]])
else:
    print("NOT_RUN: set LAB06_RUN_FULL_LIVE=1 for the controlled instruction experiment.")
"""),
    md(r"""
## 10. Типові помилки та контроль

Фактичні історичні failure modes: неправильний filename у tool arguments; пропущений calculator call; неправильний synthesis чисел; зайві повтори; endpoint/serialization errors. `SUCCESS` означає завершення SDK-циклу, а не семантичну правильність. Контроль: allowlist і validation, max_turns, timeout, trace hooks, manual correctness rubric, ізоляція sessions і окремий preflight.
"""),
    code(r"""
rows = replay_rows
errors = pd.DataFrame(rows)[["scenario_id", "status", "answer_correctness", "tool_behavior",
                             "tool_error_count", "repeated_identical_calls", "notes"]]
display(errors[(errors.answer_correctness != "CORRECT") |
               (errors.tool_error_count > 0) |
               (errors.repeated_identical_calls > 0)])
display(Markdown((LAB_ROOT / "report/methodology_audit.md").read_text(encoding="utf-8")))
"""),
    md(r"""
## 11. Збереження результатів і provenance

LIVE smoke містить run ID, UTC timestamp, model ID, endpoint без ключа, prompt, instructions, tools, final output, calls, runtime, status і trace path. REPLAY не записується як новий запуск. Історичні `outputs/runs` та `outputs/traces` не перезаписуються notebook-ом.
"""),
    code(r"""
provenance_summary = {
    "LIVE_preflight": preflight,
    "LIVE_smoke": smoke_record,
    "FULL_LIVE_matrix_enabled": RUN_FULL_LIVE,
    "REPLAY_batch": replay_batch["batch_id"],
    "REPLAY_verification": replay_verification,
}
save_json(LIVE_ROOT / "latest_notebook_summary.json", provenance_summary)
display(provenance_summary)
"""),
    md(r"""
## 12. Порівняльний аналіз і візуалізація

Числа нижче обчислені з verified historical records. Вони описують лише цю серію, не універсальну якість моделі.
"""),
    code(r"""
analysis = pd.DataFrame(rows)
summary = analysis.groupby("category", as_index=False).agg(
    runs=("run_id", "count"),
    mean_runtime_seconds=("runtime_seconds", "mean"),
    mean_tool_calls=("tool_call_count", "mean"),
)
display(summary)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
summary.plot.bar(x="category", y="mean_tool_calls", ax=axes[0], legend=False, title="Mean tool calls (REPLAY)")
summary.plot.bar(x="category", y="mean_runtime_seconds", ax=axes[1], legend=False, title="Mean runtime, s (REPLAY)")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 13. Висновки

Фактичні traces доводять, що SDK виконує structured tool calls, але сам факт виклику не гарантує правильної відповіді. Read-only tools дають grounding, SQLiteSession переносить історію між turns та ізолює незалежні session IDs, а multi-step trace показує лише спостережувану послідовність дій. Строгі instructions знижують ризик вигадування, але не усувають помилки малої локальної моделі. Основні межі: literal search, обмежене context window, пропущені або зайві calls, неправильний synthesis і недетермінованість навіть при temperature 0.

Статус обов'язкових LIVE-перевірок не слід називати FULL PASS, якщо повна матриця не запускалася або endpoint був недоступний.
"""),
    code(r"""
qa_status = {
    "endpoint_preflight": preflight["status"],
    "live_agent_smoke": smoke_record["status"],
    "full_live_matrix": "PASS" if RUN_FULL_LIVE and client is not None else "NOT_RUN",
    "replay_loaded": "PASS",
}
display(qa_status)
if client is not None:
    await client.close()
"""),
]


notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3 (Lab 06)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
)
nbformat.write(notebook, LAB_ROOT / "notebooks" / "01_openai_agents_sdk.ipynb")
print(f"Wrote {len(cells)} cells")
