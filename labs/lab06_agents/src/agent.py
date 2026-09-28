"""An explicit local model adapter prevents cloud provider fallback."""
from dataclasses import replace
import httpx2
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from .config import INSTRUCTIONS, Settings
from .tools import build_tools


async def connect(settings: Settings):
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key,
                         timeout=settings.timeout_seconds, max_retries=0,
                         http_client=httpx2.AsyncClient(trust_env=False, follow_redirects=False))
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    try:
        available = (await client.models.list()).data
        candidates = [m.id for m in available if "embedding" not in m.id.lower()]
        model_id = settings.model_id or (candidates[0] if candidates else "")
        if not model_id or model_id not in candidates:
            raise RuntimeError("No requested local chat model is available")
        return client, replace(settings, model_id=model_id)
    except BaseException:
        await client.close()
        raise


def create_coursework_agent(client, settings, with_tools=True):
    return Agent(name="Coursework Assistant", instructions=INSTRUCTIONS,
                 model=OpenAIChatCompletionsModel(settings.model_id, client),
                 tools=build_tools() if with_tools else [],
                 model_settings=ModelSettings(temperature=settings.temperature,
                     tool_choice="auto", parallel_tool_calls=False, max_tokens=settings.max_tokens))
