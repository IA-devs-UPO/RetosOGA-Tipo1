"""Orchestrator using provider abstraction (Google → OpenRouter → Ollama → others)."""
import asyncio
import os
import sys

from ai_providers import get_available_provider, create_client, get_provider_info

ORCHESTRATOR_PERSONA = """Eres el Orquestador del Reto de Inteligencia Artificial de Series Temporales de OGATHON.
El usuario te pedirá ayuda para resolver distintos desafíos matemáticos, algorítmicos o de código.

Tienes a tu disposición dos subagentes especialistas:
1. Especialista en Boosting (para el Desafío 1 sobre MAE, GBDT, Log-Cosh, Huber).
2. Especialista en Clustering (para el Desafío 2 sobre segmentación de series, DTW, k-Shape, Louvain).

Analiza la consulta del usuario, decide qué subagente está mejor capacitado para responder, y delega la tarea.
Provee al usuario una respuesta integral consolidando lo que diga tu subagente.
"""

def main():
    info = get_provider_info()
    if not info["active"]:
        print("Error: No hay proveedor de IA disponible.")
        print("Configura al menos una de estas variables:")
        print("  - GEMINI_API_KEY (Google)")
        print("  - OPENROUTER_API_KEY (OpenRouter)")
        print("  - OLLAMA_BASE_URL (Ollama local)")
        sys.exit(1)

    print(f"Orquestador OGATHON")
    print(f"Provider: {info['active']} ({info['client_type']})")
    print(f"Modelo: {info['model']}")
    print("Escribe 'salir' para terminar.\n")

    async def run():
        provider = get_available_provider()
        client = create_client(provider)

        messages = [{"role": "system", "content": ORCHESTRATOR_PERSONA}]

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