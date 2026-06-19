"""
AI Provider Abstraction Module.

Provides a factory pattern for configuring AI model providers with a centralized catalog.
Supports: google, ollama, nvidia, openrouter, custom.

Environment variables are loaded from .env file via python-dotenv.
"""
from dataclasses import dataclass
from typing import Optional
import os

from dotenv import load_dotenv

# Load environment variables from .env file at module initialization
load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI model provider."""

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


PROVIDER_CATALOG: dict[str, ProviderConfig] = {
    "google": ProviderConfig(
        model="models/gemini-2.5-pro",
        api_key=os.environ.get("GEMINI_API_KEY"),
    ),
    "ollama": ProviderConfig(
        model="ollama/gemma4:12b",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    "nvidia": ProviderConfig(
        model="nvidia/deepseek-v4-flash",
        api_key=os.environ.get("NVIDIA_API_KEY"),
    ),
    "openrouter": ProviderConfig(
        model=f"openrouter/{os.environ.get('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')}",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    ),
    "custom": ProviderConfig(
        model=os.environ.get("CUSTOM_MODEL", "gpt-4"),
        api_key=os.environ.get("CUSTOM_API_KEY"),
        base_url=os.environ.get("CUSTOM_ENDPOINT"),
    ),
}


def get_provider_config() -> ProviderConfig:
    """
    Get the provider configuration based on the PROVIDER environment variable.

    Defaults to 'google' if PROVIDER is not set.
    Validates required environment variables for the selected provider.

    Returns:
        ProviderConfig with model, api_key, and base_url.

    Raises:
        ValueError: If the provider is not supported.
        EnvironmentError: If required environment variables are missing.
    """
    provider = os.environ.get("PROVIDER", "google")

    if provider not in PROVIDER_CATALOG:
        available = ", ".join(PROVIDER_CATALOG.keys())
        raise ValueError(
            f"Provider '{provider}' no soportado. Opciones: {available}"
        )

    config = PROVIDER_CATALOG[provider]
    _validate_provider_config(provider, config)
    return config


def _validate_provider_config(provider: str, config: ProviderConfig) -> None:
    """
    Validate that required environment variables are set for the provider.

    Args:
        provider: The provider name.
        config: The provider configuration.

    Raises:
        EnvironmentError: If a required environment variable is missing.
    """
    if provider == "google" and not config.api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY no está definida. "
            "Por favor, obtenla en https://aistudio.google.com/app/api-keys"
        )
    elif provider == "nvidia" and not config.api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY no está definida para provider=nvidia"
        )
    elif provider == "openrouter" and not config.api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY no está definida para provider=openrouter"
        )
    elif provider == "custom":
        if not config.api_key:
            raise EnvironmentError(
                "CUSTOM_API_KEY no está definida para provider=custom"
            )
        if not config.base_url:
            raise EnvironmentError(
                "CUSTOM_ENDPOINT no está definida para provider=custom"
            )