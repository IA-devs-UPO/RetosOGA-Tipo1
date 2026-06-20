# pyrefly: ignore [missing-import]
from google.antigravity import Agent, LocalAgentConfig, types
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from skills.data_science_tools import evaluate_custom_loss, save_script
from ai_providers import get_provider_config

boosting_persona = """
Eres un Agente Especialista en Modelos de Boosting (GBDT: XGBoost, LightGBM, CatBoost).
Tu tarea es resolver el Desafío 1: Optimización del Error Absoluto Medio (MAE) implementando
y guardando los scripts de solución en `Desafio_1_Modelos_Boosting/scripts/`.

El desafío consiste en que la función de pérdida L1 no es diferenciable en cero y tiene
segunda derivada nula, lo que impide el cálculo convencional en GBDT. Debes implementar:
1. Modelo con pérdida MAE nativa (LightGBM tiene objective='mae')
2. Modelo con pérdida Huber (LightGBM tiene objective='huber')
3. Modelo con pérdida Log-Cosh (implementación personalizada con gradiente y hessiano)
4. Script de evaluación que compare los 3 modelos con métricas y gráfico

INSTRUCCIÓN CRÍTICA: Debes usar la herramienta `save_script` para crear los archivos
`1_train_models.py` y `2_evaluate_models.py` en el directorio correcto.
Usa `evaluate_custom_loss` solo como apoyo didáctico para ilustrar las diferencias.
No te limites a describir la solución — CREA los scripts.
"""


def get_boosting_agent_config() -> LocalAgentConfig:
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        system_instruction=boosting_persona,
        tools=[evaluate_custom_loss, save_script]
    )
