import asyncio
import os
import sys
from google.antigravity import Agent, LocalAgentConfig, types

sys.path.append(os.path.join(os.path.dirname(__file__), '.agents'))

from boosting_agent import get_boosting_agent_config
from clustering_agent import get_clustering_agent_config
from ai_providers import get_provider_config

orchestrator_persona = """
Eres el Orquestador del Reto de Inteligencia Artificial de Series Temporales de OGATHON.
El usuario te pedirá ayuda para resolver distintos desafíos matemáticos, algorítmicos o de código.

Tienes a tu disposición dos subagentes especialistas:
1. Especialista en Boosting (para el Desafío 1 sobre MAE, GBDT, Log-Cosh, Huber).
2. Especialista en Clustering (para el Desafío 2 sobre segmentación de series, DTW, k-Shape, Louvain).

Analiza la consulta del usuario, decide qué subagente está mejor capacitado para responder, y delega la tarea.
Provee al usuario una respuesta integral consolidando lo que diga tu subagente.
"""


def get_orchestrator_config() -> LocalAgentConfig:
    """Get the orchestrator configuration using the provider factory."""
    provider_cfg = get_provider_config()
    return LocalAgentConfig(
        model=provider_cfg.model,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        system_instruction=orchestrator_persona,
        capabilities=types.CapabilitiesConfig(
            enable_subagents=True,
        )
    )


def main():
    # Obtener configuración del provider (valida env vars)
    config = get_orchestrator_config()

    async def run_orchestrator():
        # Inicializamos el orquestador
        async with Agent(config) as agent:
            if len(sys.argv) > 1:
                # File mode
                file_path = sys.argv[1]
                print(f"Leyendo consulta desde {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    user_input = "Por favor, actúa sobre el siguiente enunciado y resuélvelo usando tus subagentes expertos:\n\n" + f.read()
                print("Orquestador pensando y consultando especialistas...\n")
                response = await agent.chat(user_input)
                print(f"Orquestador:\n{await response.text()}\n")
            else:
                # Interactive mode
                print("Bienvenido al Orquestador del Reto de Series Temporales (OGATHON).")
                print("Escribe 'salir' para terminar.\n")
                
                while True:
                    user_input = input("Usuario: ")
                    if user_input.lower() in ['salir', 'exit', 'quit']:
                        break
                    
                    print("Orquestador pensando y consultando especialistas...\n")
                    response = await agent.chat(user_input)
                    print(f"Orquestador: {await response.text()}\n")

    asyncio.run(run_orchestrator())

if __name__ == "__main__":
    main()
