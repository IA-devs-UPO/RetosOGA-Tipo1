# pyrefly: ignore [missing-import]
from google.antigravity import Agent, LocalAgentConfig, types
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from skills.data_science_tools import run_clustering_analysis, save_script
from ai_providers import get_provider_config

clustering_persona = """
Eres un Agente Especialista en Segmentación de Series Temporales.
Tu tarea es resolver el Desafío 2: Clustering de Series Temporales implementando
y guardando los scripts de solución en `Desafio_2_Segmentacion_Clustering/scripts/`.

El desafío consiste en clusterizar series temporales sintéticas con 3 patrones
(seno, coseno desfasado, señal cuadrada) usando 3 métodos y compararlos:
1. DTW + k-Means (TimeSeriesKMeans con metric='dtw')
2. k-Shape (KShape de tslearn)
3. Louvain (grafo de afinidad con DTW + python-louvain)

INSTRUCCIÓN CRÍTICA: Debes usar la herramienta `save_script` para crear los archivos
`1_cluster_timeseries.py` y `2_evaluate_clustering.py` en el directorio correcto.
Usa `run_clustering_analysis` para ejecutar el análisis y validar que los resultados
tienen sentido, pero NO te limites a describir — CREA los scripts completos y funcionales
con generación de datos sintéticos, ejecución de los 3 métodos, cálculo de métricas
(ARI, silhouette, modularidad) y generación de gráficos comparativos.
"""


def get_clustering_agent_config() -> LocalAgentConfig:
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        system_instruction=clustering_persona,
        tools=[run_clustering_analysis, save_script]
    )
