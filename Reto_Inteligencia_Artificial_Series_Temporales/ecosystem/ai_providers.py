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
from dataclasses import dataclass
from typing import Optional, Literal
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


class BaseAIClient(ABC):
    """Abstract base for AI clients."""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Send chat request and return response text."""
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
        extra_headers = None
        if self._is_openrouter:
            extra_headers = {
                "HTTP-Referer": "https://github.com/ogathon/retos-oga",
                "X-Title": "OGA Series Temporales",
            }
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            extra_headers=extra_headers,
            **{"max_tokens": 4096, **kwargs},
        )
        return response.choices[0].message.content


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


def get_provider_info() -> dict:
    """Get info about available providers and their status."""
    provider = get_available_provider()
    return {
        "active": provider.name if provider else None,
        "client_type": provider.client_type if provider else None,
        "model": provider.model if provider else None,
    }