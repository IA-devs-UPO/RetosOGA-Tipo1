"""
AI Provider Abstraction Module.

Factory pattern supporting multiple providers with priority:
1. Google (Gemini via Antigravity SDK)
2. OpenRouter (OpenAI-compatible)
3. Ollama (OpenAI-compatible, local)
4. Others (nvidia, custom)

Each provider uses the optimal client:
- Google: google.antigravity.Agent
- Others: openai.AsyncOpenAI (OpenAI-compatible)
"""
import inspect
import json
from dataclasses import dataclass
from typing import Optional, Literal, Callable, Any
from abc import ABC, abstractmethod
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI model provider."""

    name: str
    client_type: Literal["antigravity", "openai"]
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    priority: int = 99


def func_to_tool(fn: Callable) -> dict:
    """Convert a Python function into an OpenAI-compatible tool definition."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    params = {"type": "object", "properties": {}, "required": []}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        param_doc, param_type = "", "string"
        if doc:
            for line in doc.split("\n"):
                s = line.strip()
                if s.startswith(f"{name}:"):
                    param_doc = s[len(name) + 1:].strip()
        a = param.annotation
        if a is str: param_type = "string"
        elif a in (int, float): param_type = "number"
        elif a is bool: param_type = "boolean"
        elif getattr(a, "__origin__", None) is list: param_type = "array"
        elif getattr(a, "__origin__", None) is dict: param_type = "object"
        params["properties"][name] = {"type": param_type, "description": param_doc or name}
        if param.default is inspect.Parameter.empty:
            params["required"].append(name)
    return {"type": "function", "function": {"name": fn.__name__, "description": doc.split("\n\n")[0].strip() if doc else "", "parameters": params}}


async def run_tool(name: str, args: dict, tools_map: dict[str, Callable]) -> str:
    fn = tools_map.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = await fn(**args) if inspect.iscoroutinefunction(fn) else fn(**args)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


class BaseAIClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        pass

    async def chat_with_tools(self, messages: list[dict], tools: list[Callable], max_tool_rounds: int = 10) -> str:
        tool_defs = [func_to_tool(t) for t in tools]
        tools_map = {t.__name__: t for t in tools}
        msgs = list(messages)
        for _ in range(max_tool_rounds):
            response = await self._raw_chat(msgs, tools=tool_defs)
            choice = response.choices[0]
            msg = choice.message
            if not msg.tool_calls:
                return msg.content or ""
            am = {"role": "assistant", "content": msg.content or "", "tool_calls": []}
            for tc in msg.tool_calls:
                am["tool_calls"].append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
            msgs.append(am)
            for tc in msg.tool_calls:
                raw = tc.function.arguments
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        args = json.loads(raw.replace("'", '"'))
                    except json.JSONDecodeError:
                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": f"Invalid JSON: {raw[:200]}"})})
                        continue
                result = await run_tool(tc.function.name, args, tools_map)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return msgs[-1]["content"] if msgs else ""

    @abstractmethod
    async def _raw_chat(self, messages: list[dict], **kwargs) -> Any:
        pass


class OpenAIClient(BaseAIClient):
    """OpenAI-compatible client for OpenRouter, Ollama, etc."""

    def __init__(self, config: ProviderConfig):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )
        self._model = config.model
        self._is_openrouter = "openrouter" in config.base_url

    async def chat(self, messages: list[dict], **kwargs) -> str:
        response = await self._raw_chat(messages, **kwargs)
        return response.choices[0].message.content

    async def _raw_chat(self, messages: list[dict], **kwargs) -> Any:
        extra_headers = None
        if self._is_openrouter:
            extra_headers = {
                "HTTP-Referer": "https://github.com/ogathon/retos-oga",
                "X-Title": "OGA Series Temporales",
            }
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            extra_headers=extra_headers,
            **{"max_tokens": 4096, **kwargs},
        )


def get_available_provider() -> Optional[ProviderConfig]:
    """
    Get the first available provider based on priority.
    Priority: google > openrouter > ollama > others
    """
    providers = [
        ProviderConfig(
            name="google",
            client_type="antigravity",
            model="models/gemini-2.5-pro",
            api_key=os.environ.get("GEMINI_API_KEY"),
            priority=1,
        ),
        ProviderConfig(
            name="openrouter",
            client_type="openai",
            model=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            priority=2,
        ),
        ProviderConfig(
            name="ollama",
            client_type="openai",
            model=os.environ.get("OLLAMA_MODEL", "gemma4:12b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            priority=3,
        ),
        ProviderConfig(
            name="nvidia",
            client_type="openai",
            model=os.environ.get("NVIDIA_MODEL", "nvidia/deepseek-v4-flash"),
            api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            priority=4,
        ),
    ]

    explicit = os.environ.get("PROVIDER")
    if explicit:
        for p in providers:
            if p.name == explicit:
                if _validate_provider(p):
                    return p
        raise EnvironmentError(f"Provider '{explicit}' no disponible o sin credenciales")

    for p in sorted(providers, key=lambda x: x.priority):
        if _validate_provider(p):
            return p

    return None


def _validate_provider(config: ProviderConfig) -> bool:
    """Check if provider has required credentials."""
    if config.name == "google" and not config.api_key:
        return False
    if config.name == "openrouter" and not config.api_key:
        return False
    if config.name == "nvidia" and not config.api_key:
        return False
    if config.name == "ollama":
        return True
    return True


def create_client(config: ProviderConfig) -> BaseAIClient:
    """Factory to create the appropriate client for a provider."""
    if config.client_type == "antigravity":
        from google.antigravity import Agent, LocalAgentConfig, types

        return GoogleAntigravityClient(config)
    elif config.client_type == "openai":
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unknown client type: {config.client_type}")


class GoogleAntigravityClient(BaseAIClient):
    """Client using Google Antigravity SDK."""

    def __init__(self, config: ProviderConfig):
        from google.antigravity import Agent, LocalAgentConfig, types

        self._config = LocalAgentConfig(
            model=config.model,
            api_key=config.api_key,
        )

    async def chat(self, messages: list[dict], **kwargs) -> str:
        from google.antigravity import Agent

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        system_instruction = types.TemplatedSystemInstructions(
            sections=[types.SystemInstructionSection(content=system)]
        )

        config = self._config.model_copy()
        config.system_instructions = system_instruction

        async with Agent(config) as agent:
            response = await agent.chat("\n".join(m["content"] for m in user_messages))
            return await response.text()


def get_provider_config() -> Optional[ProviderConfig]:
    """Alias for get_available_provider. Used by sub-agents."""
    return get_available_provider()


def get_provider_info() -> dict:
    """Get info about available providers and their status."""
    provider = get_available_provider()
    return {
        "active": provider.name if provider else None,
        "client_type": provider.client_type if provider else None,
        "model": provider.model if provider else None,
    }