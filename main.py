"""Punto de entrada del agente: loop interactivo por consola.

Uso:
    python main.py

Antes de correr, asegurate de tener el ``.env`` con ``GROQ_API_KEY``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

import database
from agent import agente

# Un thread_id fijo por sesión para que la memoria funcione.
CONFIG = {"configurable": {"thread_id": "demo-1"}}

BANNER = """
==============================================
  San Cris — Asistente virtual San Cristóbal
==============================================
Escribí tu consulta (o "salir" para terminar).
Ejemplos:
  - "Hola"
  - "¿Qué plan tengo para la patente AB123CD?"
  - "Choqué hoy en la ruta con la AF987XX, ¿me cubre?"
"""


def main() -> None:
    database.inicializar_base()
    print(BANNER)

    while True:
        try:
            texto = input("\nVos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Chau!")
            break

        if not texto:
            continue
        if texto.lower() in ("salir", "exit", "chau", "quit"):
            print("¡Hasta luego!")
            break

        try:
            resultado = agente.invoke(
                {"messages": [HumanMessage(content=texto)]},
                config=CONFIG,
            )
            respuesta = resultado["messages"][-1].content
        except Exception as e:  # noqa: BLE001
            respuesta = f"Ocurrió un error: {e}"
        print(f"\nSan Cris: {respuesta}")


if __name__ == "__main__":
    main()
