# Exploration: Multi-Provider AI Abstraction

## Current State

The project uses **Google Antigravity SDK** as the core multi-agent framework. All AI connections are hardcoded to Gemini:

| File | Model | API Key |
|------|-------|---------|
| `ecosystem/orchestrator.py` | `models/gemini-2.5-pro` | `GEMINI_API_KEY` |
| `ecosystem/.agents/boosting_agent.py` | `models/gemini-2.5-pro` | `GEMINI_API_KEY` |
| `ecosystem/.agents/clustering_agent.py` | `models/gemini-2.5-pro` | `GEMINI_API_KEY` |

**Current flow**:
1. Check `GEMINI_API_KEY` env var at startup
2. Create `LocalAgentConfig` with hardcoded model string
3. Initialize `Agent` with config

## Affected Areas

- `ecosystem/orchestrator.py` — Main entry point, hardcoded model
- `ecosystem/.agents/boosting_agent.py` — Specialist agent, hardcoded model
- `ecosystem/.agents/clustering_agent.py` — Specialist agent, hardcoded model
- `ecosystem/requirements.txt` — Would need new dependencies

## Approaches

### 1. LiteLLM Integration
**Description**: Wrap Antigravity SDK with LiteLLM proxy layer
- Pros: 30+ providers, drop-in replacement, cost tracking, retries
- Cons: Additional HTTP hop, potential latency, dependency
- Effort: Medium

### 2. Custom Provider Abstraction
**Description**: Create `AIProvider` interface with provider implementations
- Pros: Full control, minimal dependencies, no extra hop
- Cons: More implementation work, manual provider updates
- Effort: High

### 3. Configuration-Driven Model Selection
**Description**: Use env vars/config to select model at runtime
- Pros: Minimal code change, works with existing SDK
- Cons: Still requires compatible models per provider
- Effort: Low

### 4. LangChain LCEL Adaptation
**Description**: Replace Antigravity with LangChain Expression Language
- Pros: Mature ecosystem, built-in provider abstraction
- Cons: Major rewrite, different agent paradigm
- Effort: Very High

## Recommendation

**Approach 1 (LiteLLM)** is recommended for this project because:
- Minimal intrusion into existing Antigravity architecture
- Supports Google, OpenAI, Ollama, Azure, Anthropic, and 25+ others
- Environment variables already used for config
- Can proxy to existing Gemini setup initially, expand later

**Hybrid approach**: Start with Approach 3 (config-driven) for quick win, then layer LiteLLM for full multi-provider support.

## Risks

- Antigravity SDK may not support custom model endpoints without modification
- Provider-specific tool/capability differences may break agent functionality
- Rate limits and quotas vary significantly between providers
- Cost tracking becomes critical with multiple providers

## Ready for Proposal

**Yes** — The exploration is complete. Next step is `sdd-propose` to define scope, approach selection, and rollback plan for the multi-provider abstraction layer.