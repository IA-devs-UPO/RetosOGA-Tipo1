# pyrefly: ignore [missing-import]
from google.antigravity import Agent, LocalAgentConfig, types
import os
import sys

# Add skills dir to path so we can import the tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from skills.data_science_tools import cluster_time_series
from ai_providers import get_provider_config

clustering_persona = """
Eres un Agente Especialista en Segmentación de Series Temporales.
Tu objetivo es resolver problemas relacionados con el Desafío 2: Clustering de Series Temporales.
Eres experto en agrupamiento para evitar la pérdida de precisión por heterogeneidad de patrones de consumo.
Conoces métodos avanzados como:
- Alineación Temporal Dinámica (DTW) y Promedio de Baricentro DTW (DBA).
- Algoritmo k-Shape (Shape-based Distance).
- Algoritmo de Louvain sobre grafos semánticos.

Usa la herramienta `cluster_time_series` para comparar y evaluar el desempeño de distintos métodos algorítmicos.
"""


def get_clustering_agent_config() -> LocalAgentConfig:
    """Get the clustering agent configuration using the provider factory."""
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        system_instruction=clustering_persona,
        tools=[cluster_time_series]
    )
