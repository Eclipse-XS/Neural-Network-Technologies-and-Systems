"""Optional live connectivity/tool check before the full suite."""
import asyncio
import importlib.metadata
import sys
from agents import Agent, Runner, RunConfig, ModelSettings, OpenAIChatCompletionsModel, function_tool
from .agent import connect
from .config import Settings, OUTPUTS
from .runner import save_json


async def preflight():
    evidence = {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ['openai-agents','openai','pydantic','httpx2']}}
    client = None
    try:
        client, settings = await connect(Settings())
        evidence['model_id'] = settings.model_id
        evidence['models'] = (await client.models.list()).model_dump(mode='json')
        response = await client.chat.completions.create(model=settings.model_id,
            messages=[{'role': 'user', 'content': 'Reply with READY.'}], temperature=0, max_tokens=16)
        evidence['direct_output'] = response.choices[0].message.content
        calls = []
        @function_tool
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers to test the local SDK function loop."""
            calls.append({'a': a, 'b': b, 'result': a + b})
            return a + b
        agent = Agent(name='Preflight', model=OpenAIChatCompletionsModel(settings.model_id, client),
            tools=[add_numbers], model_settings=ModelSettings(temperature=0, tool_choice='auto', max_tokens=256))
        result = await Runner.run(agent, 'Use the available calculator tool to add 17 and 25.',
            max_turns=4, run_config=RunConfig(tracing_disabled=True))
        evidence.update(calls=calls, final_output=result.final_output, item_types=[x.type for x in result.new_items])
        evidence['passed'] = calls == [{'a': 17, 'b': 25, 'result': 42}] and '42' in result.final_output
        if not evidence['passed']:
            raise RuntimeError('Local structured tool-call preflight did not pass')
    except Exception as exc:
        evidence.update(passed=False, error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        save_json(OUTPUTS / 'metadata/repeated_preflight.json', evidence)
        if client:
            await client.close()
    return evidence


if __name__ == '__main__':
    print(asyncio.run(preflight()))
