# pyrefly: ignore [missing-import]
from google.antigravity import Agent, LocalAgentConfig, types
import os
import sys

# Add skills dir to path so we can import the tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from skills.data_science_tools import evaluate_custom_loss

boosting_persona = """
Eres un Agente Especialista en Modelos de Boosting, específicamente GBDT (XGBoost, LightGBM, CatBoost).
Tu objetivo es resolver problemas relacionados con el Desafío 1: Optimización del Error Absoluto Medio (MAE).
Eres experto en matemáticas y conoces a fondo por qué la función $L_1$ no es diferenciable en cero y tiene segunda derivada nula.
Puedes sugerir soluciones como:
- Búsqueda Lineal y Actualización por Cuantiles
- Pérdida Huber
- Pérdida Log-Cosh

Usa la herramienta `evaluate_custom_loss` para mostrar al usuario cómo difieren las pérdidas.
"""

def get_boosting_agent_config() -> LocalAgentConfig:
    return LocalAgentConfig(
        model="models/gemini-2.5-pro",
        system_instruction=boosting_persona,
        tools=[evaluate_custom_loss]
    )
