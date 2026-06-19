# Proposal: Abstraer conexiones a servicios de IA para múltiples proveedores

## Intent

El proyecto actualmente tiene conexiones AI **hardcodeadas** a `models/gemini-2.5-pro` con `GEMINI_API_KEY` en todos los agentes. Esta propuesta introduce una capa de abstracción que permite switchear entre proveedores de IA (Google, Ollama, NVIDIA, custom) mediante la variable de entorno `PROVIDER`, manteniendo backward compatibility completa con la configuración actual.

## Scope

### In Scope
- Crear catálogo de providers preconfigurados (google, ollama, nvidia, custom)
- Implementar variable `PROVIDER` para selección dinámica
- Mantener `GEMINI_API_KEY` como default cuando `PROVIDER=google`
- Actualizar `orchestrator.py`, `boosting_agent.py`, `clustering_agent.py`
- Agregar dependencias LiteLLM al `requirements.txt`

### Out of Scope
- Migración a otro framework de agentes
- Implementar cost tracking o rate limiting
- Testing automatizado (no hay framework actual)

## Capabilities

### New Capabilities
- `ai-provider-abstraction`: Sistema de selección de proveedor de IA mediante variable de entorno
- `provider-catalog`: Catálogo preconfigurado con google, ollama, nvidia, custom

### Modified Capabilities
- None (es extensión de funcionalidad existente, no cambio de comportamiento)

## Approach

1. **LiteLLM como proxy**: Usar LiteLLM para abstraer la conexión a múltiples providers con interfaz unificada
2. **Provider catalog**: Definir configuración por provider (modelo, endpoint, API key)
3. **Env var selection**: `PROVIDER=google|ollama|nvidia|custom` selecciona el provider
4. **Default behavior**: Si `PROVIDER` no está definido, usar `google` por defecto
5. **Backward compatibility**: Si `GEMINI_API_KEY` existe y no hay `PROVIDER`, comportarse como antes

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `ecosystem/orchestrator.py` | Modified | Usar provider abstraction en vez de hardcoded model |
| `ecosystem/.agents/boosting_agent.py` | Modified | Usar provider abstraction |
| `ecosystem/.agents/clustering_agent.py` | Modified | Usar provider abstraction |
| `ecosystem/requirements.txt` | Modified | Agregar `litellm` |
| `ecosystem/AGENTS.md` | Modified | Documentar nuevas variables de entorno |

## Configuration Schema

```bash
# Selector principal (default: google)
export PROVIDER=google|ollama|nvidia|custom

# Provider: google (default si no se especifica PROVIDER)
export GEMINI_API_KEY=...

# Provider: ollama
export OLLAMA_BASE_URL=http://localhost:11434

# Provider: nvidia
export NVIDIA_API_KEY=...

# Provider: custom
export CUSTOM_API_KEY=...
export CUSTOM_ENDPOINT=https://...
```

## Provider Catalog

| Provider | Model | Endpoint | API Key |
|----------|-------|----------|---------|
| `google` | `gemini-2.5-pro` | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| `ollama` | `gemma4:12b` | `localhost:11434` | None (local) |
| `nvidia` | `deepseek-v4-flash` | `api.nvidia.com` | `NVIDIA_API_KEY` |
| `custom` | configurable | `CUSTOM_ENDPOINT` | `CUSTOM_API_KEY` |

## Default Behavior

1. Si `PROVIDER` está definido → usar ese provider
2. Si `PROVIDER` NO está definido → default a `google`
3. Validar que las variables requeridas existan antes de iniciar

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| LiteLLM introduce latencia adicional | Medium | Benchmark antes/después; permitir bypass si es crítico |
| Provider-specific tool support varía | High | Documentar limitaciones por provider; fallback a Google |
| breaking change si se renombra `GEMINI_API_KEY` | Low | Mantener variable original; solo agregar nuevas |

## Rollback Plan

1. Revertir cambios en `requirements.txt` (quitar `litellm`)
2. Restaurar strings hardcoded `models/gemini-2.5-pro` en los 3 archivos de agente
3. Eliminar archivo de abstracción `provider_config.py`
4. **No se requiere migración de datos** — solo configuración

## Dependencies

- `litellm>=1.0.0` — Multi-provider abstraction layer

## Success Criteria

- [ ] `PROVIDER=google` + `GEMINI_API_KEY` funciona igual que antes
- [ ] `PROVIDER=ollama` conecta a localhost:11434 sin API key
- [ ] `PROVIDER=nvidia` usa `NVIDIA_API_KEY` con DeepSeek
- [ ] `PROVIDER=custom` permite endpoint y key configurables
- [ ] Sin `PROVIDER` definido, usa Google por defecto
- [ ] Documentación actualizada en `AGENTS.md`