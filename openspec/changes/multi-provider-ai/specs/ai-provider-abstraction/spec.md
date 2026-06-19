# Delta Spec: AI Provider Abstraction

## ADDED Requirements

### Requirement: Provider Selection via Environment Variable

The system SHALL support dynamic AI provider selection via the `PROVIDER` environment variable. When `PROVIDER` is unset, the system SHALL default to `google`.

#### Scenario: Provider Selection with Valid Provider

- GIVEN `PROVIDER=ollama` is set and `OLLAMA_BASE_URL` is configured
- WHEN the orchestrator initializes
- THEN the system SHALL connect to the Ollama endpoint at `localhost:11434`

#### Scenario: Default Provider Fallback

- GIVEN no `PROVIDER` environment variable is set
- WHEN the orchestrator initializes
- THEN the system SHALL default to `google` provider and use `GEMINI_API_KEY`

### Requirement: Provider Catalog

The system MUST provide a preconfigured catalog of AI providers:

| Provider | Model | Endpoint | API Key Required |
|----------|-------|----------|------------------|
| `google` | `gemini-2.5-pro` | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| `ollama` | `gemma4:12b` | `localhost:11434` | None |
| `nvidia` | `deepseek-v4-flash` | `api.nvidia.com` | `NVIDIA_API_KEY` |
| `openrouter` | configurable | `openrouter.ai` | `OPENROUTER_API_KEY` |
| `custom` | configurable | `CUSTOM_ENDPOINT` | `CUSTOM_API_KEY` |

#### Scenario: Google Provider with Existing Config

- GIVEN `GEMINI_API_KEY` is set and `PROVIDER` is either unset or set to `google`
- WHEN agents initialize
- THEN the system SHALL use `models/gemini-2.5-pro` exactly as before

#### Scenario: Ollama Local Provider

- GIVEN `PROVIDER=ollama` and Ollama service is running locally
- WHEN agents initialize
- THEN the system SHALL connect to `localhost:11434` without requiring any API key

#### Scenario: NVIDIA Provider

- GIVEN `PROVIDER=nvidia` and `NVIDIA_API_KEY` is set
- WHEN agents initialize
- THEN the system SHALL use the NVIDIA API endpoint with DeepSeek model

#### Scenario: Custom Provider

- GIVEN `PROVIDER=custom`, `CUSTOM_API_KEY`, and `CUSTOM_ENDPOINT` are set
- WHEN agents initialize
- THEN the system SHALL use the custom endpoint with provided credentials

### Requirement: Configuration Validation

The system SHALL validate required environment variables before agent initialization. If required variables are missing for the selected provider, the system MUST emit a descriptive error and halt.

#### Scenario: Missing API Key for Selected Provider

- GIVEN `PROVIDER=nvidia` is set but `NVIDIA_API_KEY` is not
- WHEN the orchestrator initializes
- THEN the system SHALL print error: "NVIDIA_API_KEY no está definida para provider=nvidia"
- AND SHALL terminate execution

### Requirement: Single-Variable Provider Switching

The system SHALL allow switching between providers by ONLY changing the `PROVIDER` variable. No code changes SHALL be required to switch providers.

#### Scenario: Switching from Google to Ollama

- GIVEN the system is running with `PROVIDER=google` and `GEMINI_API_KEY`
- WHEN `PROVIDER` is changed to `ollama` and the system is restarted
- THEN the system SHALL use Ollama without any other configuration changes

## MODIFIED Requirements

### Requirement: Agent Configuration Initialization

The system SHALL obtain model configuration from a provider abstraction layer instead of hardcoded model strings. Each agent config function (orchestrator, boosting, clustering) SHALL call `get_provider_config()` to retrieve the appropriate `LocalAgentConfig`.
(Previously: `LocalAgentConfig` was created with hardcoded `model="models/gemini-2.5-pro"`)

#### Scenario: Orchestrator Uses Dynamic Provider

- GIVEN `PROVIDER=google` is set with valid `GEMINI_API_KEY`
- WHEN `orchestrator.py` runs
- THEN `get_orchestrator_config()` SHALL return `LocalAgentConfig` with `model="models/gemini-2.5-pro"`

#### Scenario: Boosting Agent Uses Dynamic Provider

- GIVEN `PROVIDER=ollama` is set with local Ollama running
- WHEN `get_boosting_agent_config()` is called
- THEN the returned config SHALL use `gemma4:12b` model from Ollama

## REMOVED Requirements

None — all existing behavior is preserved via the `google` default provider.

## Configuration Schema

```bash
# Required for all providers
export PROVIDER=google|ollama|nvidia|openrouter|custom  # default: google

# Google (default)
export GEMINI_API_KEY=...

# Ollama (local)
export OLLAMA_BASE_URL=http://localhost:11434  # optional, defaults to localhost

# NVIDIA
export NVIDIA_API_KEY=...

# OpenRouter
export OPENROUTER_API_KEY=...

# Custom
export CUSTOM_API_KEY=...
export CUSTOM_ENDPOINT=https://your-custom-endpoint.com/v1
```

## Acceptance Criteria

- [ ] `PROVIDER=google` + `GEMINI_API_KEY` produces identical behavior to current hardcoded setup
- [ ] `PROVIDER=ollama` connects to `localhost:11434` without API key
- [ ] `PROVIDER=nvidia` uses `NVIDIA_API_KEY` with DeepSeek model
- [ ] `PROVIDER=openrouter` uses `OPENROUTER_API_KEY` with configurable model
- [ ] `PROVIDER=custom` allows fully custom endpoint and API key
- [ ] Missing required env vars produce clear error message with provider name
- [ ] All three agents (orchestrator, boosting, clustering) use the same provider