# AGENTS.md

## Setup commands
- Instalar dependencias: `pip install -r requirements.txt`
- Configurar variables de entorno: `export GEMINI_API_KEY="your-api-key"`
- Ejecutar orquestador: `python orchestrator.py`

## Architecture & Agent Definitions
Este repositorio contiene un ecosistema de agentes basado en el Google Antigravity SDK diseñado para resolver retos de Data Science en series temporales.

- **Orquestador Principal**: `orchestrator.py` (Enrutador de solicitudes).
- **Subagentes**: Ubicados en el directorio oculto `.agents/`.
  - `boosting_agent.py`: Experto en GBDT y métricas sustitutas del MAE.
  - `clustering_agent.py`: Experto en Segmentación, DTW, y Grafos Semánticos.
- **Skills/Tools**: Ubicados en `skills/`, proveen código Python simluado para análisis numérico y operaciones matemáticas complejas.

## Code style
- Usar imports relativos añadiendo al `sys.path` cuando sea necesario para acceder a `.agents/`.
- Tipado estricto.
