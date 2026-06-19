# AGENTS.md

## Setup commands
- Instalar dependencias: `pip install -r requirements.txt`
- Configurar variables de entorno (ver sección de configuración de proveedores)
- Ejecutar orquestador: `python orchestrator.py`

## Provider Configuration

### Variable Principal
Establece el proveedor de IA mediante la variable de entorno `PROVIDER`.

```bash
export PROVIDER=google  # Valor por defecto
```

### Catálogo de Proveedores

| Provider | Modelo | Variable de Entorno | Notas |
|----------|--------|---------------------|-------|
| google | gemini-2.5-pro | `GEMINI_API_KEY` | Por defecto |
| ollama | gemma4:12b | `OLLAMA_BASE_URL` | Local (localhost:11434) |
| nvidia | deepseek-v4-flash | `NVIDIA_API_KEY` | API NVIDIA |
| openrouter | configurable | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Modelo por defecto: claude-3.5-sonnet |
| custom | configurable | `CUSTOM_API_KEY`, `CUSTOM_ENDPOINT`, `CUSTOM_MODEL` | Endpoint OpenAI-compatible |

### Ejemplos de Configuración

**Google (por defecto):**
```bash
export PROVIDER=google
export GEMINI_API_KEY="tu-api-key"
```

**Ollama (local):**
```bash
export PROVIDER=ollama
export OLLAMA_BASE_URL="http://localhost:11434"
```

**NVIDIA:**
```bash
export PROVIDER=nvidia
export NVIDIA_API_KEY="tu-nvidia-api-key"
```

**OpenRouter:**
```bash
export PROVIDER=openrouter
export OPENROUTER_API_KEY="tu-openrouter-key"
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"  # Opcional
```

**Custom (OpenAI-compatible):**
```bash
export PROVIDER=custom
export CUSTOM_API_KEY="tu-api-key"
export CUSTOM_ENDPOINT="https://tu-servicio.openai.com/v1"
export CUSTOM_MODEL="gpt-4"  # Opcional
```

## Architecture & Agent Definitions
Este repositorio contiene un ecosistema de agentes basado en el Google Antigravity SDK diseñado para resolver retos de Data Science en series temporales.

- **Orquestador Principal**: `orchestrator.py` (Enrutador de solicitudes).
- **Subagentes**: Ubicados en el directorio oculto `.agents/`.
  - `boosting_agent.py`: Experto en GBDT y métricas sustitutas del MAE.
  - `clustering_agent.py`: Experto en Segmentación, DTW, y Grafos Semánticos.
- **Skills/Tools**: Ubicados en `skills/`, proveen código Python simluado para análisis numérico y operaciones matemáticas complejas.
- **Provider Factory**: `ai_providers.py` (Abstracción de proveedores de IA).

## Code style
- Usar imports relativos añadiendo al `sys.path` cuando sea necesario para acceder a `.agents/`.
- Tipado estricto.
- Usar `get_provider_config()` para obtener la configuración del proveedor activo.
