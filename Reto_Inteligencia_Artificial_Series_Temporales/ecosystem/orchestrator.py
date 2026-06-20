"""Orchestrator using provider abstraction (Google → OpenRouter → Ollama → others)."""
import argparse
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.agents'))

from ai_providers import get_available_provider, create_client, get_provider_info
from skills.data_science_tools import (
    evaluate_custom_loss, run_clustering_analysis, save_script,
)

agents_dir = os.path.join(os.path.dirname(__file__), '.agents')
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

import importlib
boosting_mod = importlib.import_module('boosting_agent')
clustering_mod = importlib.import_module('clustering_agent')
boosting_persona = boosting_mod.boosting_persona
clustering_persona = clustering_mod.clustering_persona

ORCHESTRATOR_PERSONA = """Eres el Orquestador del Reto de Inteligencia Artificial de Series Temporales de OGATHON.
El usuario te pedirá ayuda para resolver distintos desafíos matemáticos, algorítmicos o de código.

Tienes a tu disposición dos subagentes especialistas:
1. Especialista en Boosting (para el Desafío 1 sobre MAE, GBDT, Log-Cosh, Huber).
2. Especialista en Clustering (para el Desafío 2 sobre segmentación de series, DTW, k-Shape, Louvain).

Analiza la consulta del usuario, decide qué subagente está mejor capacitado para responder, y delega la tarea.
Provee al usuario una respuesta integral consolidando lo que diga tu subagente.
"""

AGENT_CONFIGS = {
    "clustering": {
        "persona": clustering_persona,
        "tools": [run_clustering_analysis, save_script],
    },
    "boosting": {
        "persona": boosting_persona,
        "tools": [evaluate_custom_loss, save_script],
    },
}


async def run_agent_via_llm(agent_name: str):
    """Run an agent using the LLM with tool calling, so it can create scripts."""
    config = AGENT_CONFIGS.get(agent_name)
    if not config:
        print(f"Agente desconocido: {agent_name}")
        print("Agentes disponibles: " + ", ".join(AGENT_CONFIGS.keys()))
        return

    info = get_provider_info()
    if not info["active"]:
        print("Error: No hay proveedor de IA disponible.")
        return

    provider = get_available_provider()
    client = create_client(provider)

    desafios = {
        "clustering": "Desafio_2_Segmentacion_Clustering",
        "boosting": "Desafio_1_Modelos_Boosting",
    }
    desafio_dir = desafios.get(agent_name, f"Desafio_{agent_name}")
    enunciado_path = os.path.join(os.path.dirname(__file__), "..", desafio_dir, "enunciado.md")

    mensaje = f"Resuelve el desafío completando TODAS las tareas indicadas en tu persona."
    if os.path.exists(enunciado_path):
        with open(enunciado_path, "r", encoding="utf-8") as f:
            mensaje = f"Resuelve el siguiente desafío creando los scripts necesarios. Usa save_script para escribir cada archivo en el directorio correcto. Luego ejecuta el análisis para validar los resultados.\n\nENUNCIADO:\n{f.read()}"

    messages = [
        {"role": "system", "content": config["persona"]},
        {"role": "user", "content": mensaje},
    ]

    print(f"\n=== {agent_name.title()} Agent trabajando... ===\n")
    response = await client.chat_with_tools(messages, tools=config["tools"])
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Orquestador OGATHON")
    parser.add_argument("--agent", "-a", help="Ejecutar un agente directamente (clustering, boosting)")
    parser.add_argument("file", nargs="?", help="Archivo de enunciado para modo archivo")
    args = parser.parse_args()

    info = get_provider_info()
    if not info["active"]:
        print("Error: No hay proveedor de IA disponible.")
        print("Configura al menos una de estas variables:")
        print("  - GEMINI_API_KEY (Google)")
        print("  - OPENROUTER_API_KEY (OpenRouter)")
        print("  - OLLAMA_BASE_URL (Ollama local)")
        sys.exit(1)

    if args.agent:
        asyncio.run(run_agent_via_llm(args.agent))
        return

    print(f"Orquestador OGATHON")
    print(f"Provider: {info['active']} ({info['client_type']})")
    print(f"Modelo: {info['model']}")
    print("Escribe 'salir' para terminar.\n")

    async def run():
        provider = get_available_provider()
        client = create_client(provider)

        messages = [{"role": "system", "content": ORCHESTRATOR_PERSONA}]

        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
            messages.append({"role": "user", "content": f"Por favor, actúa sobre el siguiente enunciado y resuélvelo usando tus subagentes expertos:\n\n{content}"})
            print("Orquestador pensando...\n")
            response = await client.chat(messages)
            print(f"Orquestador:\n{response}\n")
            return

        while True:
            user_input = input("Usuario: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break

            messages.append({"role": "user", "content": user_input})
            print("Orquestador pensando...\n")

            response = await client.chat(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Orquestador: {response}\n")

    asyncio.run(run())


if __name__ == "__main__":
    main()