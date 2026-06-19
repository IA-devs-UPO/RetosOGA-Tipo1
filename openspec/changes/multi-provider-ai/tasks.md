# Tasks: AI Provider Abstraction

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~150-180 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

## Phase 1: Foundation (Provider Factory)

- [x] 1.1 Create `Reto_Inteligencia_Artificial_Series_Temporales/ecosystem/ai_providers.py` with `ProviderConfig` dataclass, `PROVIDER_CATALOG` dict, `get_provider_config()` function, and `_validate_provider_config()` helper
- [x] 1.2 Implement all 5 providers: google, ollama, nvidia, openrouter, custom with model strings and env var resolution

## Phase 2: Core Implementation (Agent Updates)

- [x] 2.1 Modify `orchestrator.py`: Remove hardcoded `model="models/gemini-2.5-pro"`, import `get_provider_config`, call it in `main()`, pass `api_key` and `base_url` to `LocalAgentConfig`
- [x] 2.2 Modify `boosting_agent.py`: Import `get_provider_config`, update `get_boosting_agent_config()` to use provider config
- [x] 2.3 Modify `clustering_agent.py`: Import `get_provider_config`, update `get_clustering_agent_config()` to use provider config

## Phase 3: Dependencies & Documentation

- [x] 3.1 Add `openai` to `requirements.txt` (required for custom/openai-compatible endpoints)
- [x] 3.2 Update `AGENTS.md`: Document `PROVIDER` env var, all provider configurations, and provider catalog table

## Phase 4: Environment Configuration (python-dotenv Integration)

- [x] 4.1 Add `python-dotenv` to `requirements.txt`
- [x] 4.2 Update `ai_providers.py` to call `load_dotenv()` at module initialization
- [x] 4.3 Create `.env.example` with all provider configurations documented

## Implementation Order

1. **ai_providers.py first** — all other tasks depend on it
2. **Agent files in parallel** — orchestrator, boosting_agent, clustering_agent are independent of each other
3. **requirements.txt and AGENTS.md last** — documentation and deps after code changes

## Files Summary

| File | Action | Lines |
|------|--------|-------|
| `ai_providers.py` | Create | ~90 |
| `orchestrator.py` | Modify | ~15 |
| `boosting_agent.py` | Modify | ~8 |
| `clustering_agent.py` | Modify | ~8 |
| `requirements.txt` | Modify | +1 |
| `AGENTS.md` | Modify | +25 |
| **Total** | | **~147** |