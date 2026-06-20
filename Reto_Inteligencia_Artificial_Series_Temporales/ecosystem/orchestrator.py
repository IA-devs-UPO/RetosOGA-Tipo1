"""Orchestrator using provider abstraction (Google → OpenRouter → Ollama → others)."""
import argparse
import asyncio
import importlib
import os
import sys

agents_dir = os.path.join(os.path.dirname(__file__), '.agents')
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from ai_providers import get_available_provider, create_client, get_provider_info

ORCHESTRATOR_PERSONA = """Eres el Orquestador del Reto de Inteligencia Artificial de Series Temporales de OGATHON.
El usuario te pedirá ayuda para resolver distintos desafíos matemáticos, algorítmicos o de código.

Tienes a tu disposición tres subagentes especialistas:
1. Especialista en Boosting (para el Desafío 1 sobre MAE, GBDT, Log-Cosh, Huber).
2. Especialista en Clustering (para el Desafío 2 sobre segmentación de series, DTW, k-Shape, Louvain).
3. Especialista en Demanda (para el Desafío 3 sobre predicción de demanda energética horaria).

Analiza la consulta del usuario, decide qué subagente está mejor capacitado para responder, y delega la tarea.
Provee al usuario una respuesta integral consolidando lo que diga tu subagente.
"""

AGENT_CONFIGS = {
    "demanda": {
        "module": "demanda_agent",
        "persona_attr": "demanda_persona",
        "tools_attr": None,
    },
}

DESAFIO_DIRS = {
    "demanda": "Desafio_3_Prediccion_Demanda",
}


async def run_agent_via_llm(agent_name: str):
    cfg = AGENT_CONFIGS.get(agent_name)
    if not cfg:
        print(f"Agente desconocido: {agent_name}")
        return

    mod = importlib.import_module(cfg["module"])
    persona = getattr(mod, cfg["persona_attr"])
    tools = getattr(mod, cfg["tools_attr"]) if cfg["tools_attr"] else None
    if tools is None:
        from skills.data_science_tools import explore_demand_data, run_demand_pipeline, save_script
        tools = [explore_demand_data, run_demand_pipeline, save_script]

    info = get_provider_info()
    if not info["active"]:
        print("Error: No hay proveedor de IA disponible.")
        return

    provider = get_available_provider()
    client = create_client(provider)
    desafio_dir = DESAFIO_DIRS.get(agent_name, f"Desafio_{agent_name}")
    enunciado_path = os.path.join(os.path.dirname(__file__), "..", desafio_dir, "enunciado.md")

    if os.path.exists(enunciado_path):
        with open(enunciado_path, "r", encoding="utf-8") as f:
            mensaje = f"Resuelve el siguiente desafío. Primero explora los datos con explore_demand_data(), luego ejecuta el pipeline con run_demand_pipeline(), y finalmente crea los scripts con save_script().\n\nENUNCIADO:\n{f.read()}"
    else:
        mensaje = f"Resuelve el desafío {agent_name} usando las herramientas disponibles."

    messages = [{"role": "system", "content": persona}, {"role": "user", "content": mensaje}]
    print(f"\n=== {agent_name.title()} Agent trabajando... ===\n")
    response = await client.chat_with_tools(messages, tools=tools)
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Orquestador OGATHON")
    parser.add_argument("--agent", "-a", help="Ejecutar agente directo (demanda)")
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