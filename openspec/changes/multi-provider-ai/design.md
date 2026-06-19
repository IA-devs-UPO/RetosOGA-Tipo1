# Design: AI Provider Abstraction

## Technical Approach

Implement a **Factory Pattern** with a **Provider Catalog** to abstract AI model connections. The `provider_config.py` module will centralize provider selection logic, validate environment variables, and return the appropriate model configuration for each provider.

**Mapping to Proposal**: LiteLLM approach is NOT needed — the Google Antigravity SDK already supports multi-provider via its own abstraction. We only need to parameterize the `model` string and API key handling.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        orchestrator.py                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  get_orchestrator_config()                               │   │
│  │    └─► ai_providers.py::get_provider_config()            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ai_providers.py                             │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ ProviderCatalog  │  │ get_provider_   │                    │
│  │ (dataclass)      │  │ config()        │                    │
│  └──────────────────┘  └────────┬────────┘                    │
│                                  │                               │
│         ┌────────────────────────┼────────────────────────┐    │
│         ▼                        ▼                        ▼    │
│  ┌─────────────┐         ┌─────────────┐          ┌──────────┐ │
│  │   google    │         │   ollama    │          │  nvidia  │ │
│  │ gemini-2.5  │         │ gemma4:12b  │          │deepseek  │ │
│  └─────────────┘         └─────────────┘          └──────────┘ │
│                                                                 │
│  ┌─────────────┐         ┌─────────────┐                      │
│  │ openrouter  │         │   custom    │                      │
│  │ configurable│         │ configurable│                      │
│  └─────────────┘         └─────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              .agents/boosting_agent.py                          │
│  get_boosting_agent_config() ──► ai_providers.py::get_provider_config()
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│             .agents/clustering_agent.py                         │
│  get_clustering_agent_config() ──► ai_providers.py::get_provider_config()
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Environment Variables
        │
        ▼
┌───────────────────┐
│ PROVIDER=google   │ (default if unset)
└─────────┬─────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ ai_providers.py::get_provider_config()  │
│   1. Validate required env vars        │
│   2. Build model string (provider-specific)
│   3. Return (model, api_key, endpoint) │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ LocalAgentConfig(                       │
│   model="{resolved_model}",             │
│   system_instruction=...,               │
│   tools=...                            │
│ )                                       │
└─────────────────────────────────────────┘
```

## Architecture Decisions

### Decision 1: Factory Pattern vs LiteLLM

| Option | Tradeoff | Decision |
|--------|----------|----------|
| LiteLLM proxy | Adds dependency, potential latency | **REJECTED** — Google Antigravity SDK already abstracts providers |
| Factory with Provider Catalog | Minimal deps, direct SDK usage | **CHOSEN** — Follows existing pattern, no extra layer |

**Rationale**: The proposal mentioned LiteLLM, but the Google Antigravity SDK already supports multiple providers through its model string format. Adding LiteLLM would be redundant.

### Decision 2: Provider Model String Format

| Provider | Model String Format |
|----------|---------------------|
| google | `models/gemini-2.5-pro` |
| ollama | `ollama/gemma4:12b` |
| nvidia | `nvidia/deepseek-v4-flash` |
| openrouter | `openrouter/{OPENROUTER_MODEL:-anthropic/claude-3.5-sonnet}` |
| custom | `openai/{CUSTOM_MODEL:-gpt-4}` with `base_url={CUSTOM_ENDPOINT}` |

**Rationale**: Google Antigravity SDK uses `{provider}/{model}` format matching LiteLLM conventions.

### Decision 3: Environment Variable Validation

**Choice**: Fail-fast with descriptive errors at initialization time.

**Alternatives**: Lazy validation (at first call) — rejected because agents should fail immediately if misconfigured.

## Provider Factory Implementation

```python
# ecosystem/ai_providers.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass(frozen=True)
class ProviderConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

PROVIDER_CATALOG = {
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
        model=f"{os.environ.get('CUSTOM_MODEL', 'gpt-4')}",
        api_key=os.environ.get("CUSTOM_API_KEY"),
        base_url=os.environ.get("CUSTOM_ENDPOINT"),
    ),
}

def get_provider_config() -> ProviderConfig:
    provider = os.environ.get("PROVIDER", "google")
    if provider not in PROVIDER_CATALOG:
        raise ValueError(f"Provider '{provider}' no soportado. Opciones: {list(PROVIDER_CATALOG.keys())}")
    
    config = PROVIDER_CATALOG[provider]
    _validate_provider_config(provider, config)
    return config

def _validate_provider_config(provider: str, config: ProviderConfig) -> None:
    """Validate required env vars for the selected provider."""
    if provider == "google" and not config.api_key:
        raise EnvironmentError("GEMINI_API_KEY no está definida para provider=google")
    if provider == "nvidia" and not config.api_key:
        raise EnvironmentError("NVIDIA_API_KEY no está definida para provider=nvidia")
    if provider == "openrouter" and not config.api_key:
        raise EnvironmentError("OPENROUTER_API_KEY no está definida para provider=openrouter")
    if provider == "custom":
        if not config.api_key:
            raise EnvironmentError("CUSTOM_API_KEY no está definida para provider=custom")
        if not config.base_url:
            raise EnvironmentError("CUSTOM_ENDPOINT no está definida para provider=custom")
```

## Integration Points

### orchestrator.py

```python
# Before (hardcoded)
config = LocalAgentConfig(
    model="models/gemini-2.5-pro",
    ...
)

# After
from ai_providers import get_provider_config

def get_orchestrator_config() -> LocalAgentConfig:
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        system_instruction=orchestrator_persona,
        capabilities=types.CapabilitiesConfig(enable_subagents=True),
    )
```

### boosting_agent.py / clustering_agent.py

```python
# Before (hardcoded)
def get_boosting_agent_config() -> LocalAgentConfig:
    return LocalAgentConfig(
        model="models/gemini-2.5-pro",
        ...
    )

# After
from ai_providers import get_provider_config

def get_boosting_agent_config() -> LocalAgentConfig:
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        ...
    )
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `ecosystem/ai_providers.py` | **Create** | Provider factory with catalog and validation |
| `ecosystem/orchestrator.py` | Modify | Replace hardcoded model with `get_provider_config()` |
| `ecosystem/.agents/boosting_agent.py` | Modify | Replace hardcoded model with `get_provider_config()` |
| `ecosystem/.agents/clustering_agent.py` | Modify | Replace hardcoded model with `get_provider_config()` |
| `ecosystem/requirements.txt` | Modify | Add `openai` (required for custom/openai-compatible endpoints) |
| `ecosystem/AGENTS.md` | Modify | Document new environment variables |
| `openspec/changes/multi-provider-ai/design.md` | Create | This design document |

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Provider catalog resolution | Direct function calls with env vars |
| Unit | Validation errors | Mock missing env vars, assert exceptions |
| Integration | End-to-end with each provider | Manual testing with real API keys |

**Note**: No automated test framework detected in project. Manual verification per acceptance criteria.

## Migration / Rollback

**No data migration required** — this is purely configuration-driven.

### Migration Path (Backward Compatibility)

1. Existing deployments with `GEMINI_API_KEY` set work unchanged (PROVIDER defaults to `google`)
2. New deployments can set `PROVIDER=ollama` for local development
3. No breaking changes to existing code paths

### Rollback Plan

1. Revert changes to `orchestrator.py`, `boosting_agent.py`, `clustering_agent.py`
2. Restore hardcoded `model="models/gemini-2.5-pro"` strings
3. Delete `ai_providers.py`
4. Remove `openai` from `requirements.txt` if added

## Open Questions

- [ ] Should `OPENROUTER_MODEL` default to a specific model or require explicit setting?
- [ ] Should we add provider-specific rate limiting or retry logic?
- [ ] Do we need to document which providers support subagents?