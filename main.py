"""Punto de entrada del agente: loop interactivo por consola.

Uso:
    python main.py

Antes de correr, asegurate de tener el ``.env`` con ``GROQ_API_KEY``.
"""

from __future__ import annotations

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

import costos
import database
import llm
from agent import agente, mensaje_para_el_usuario

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
    total = costos.Consumo()

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

        # Suma los tokens de todas las llamadas al LLM de este turno.
        contador = UsageMetadataCallbackHandler()
        try:
            resultado = agente.invoke(
                {"messages": [HumanMessage(content=texto)]},
                config={**CONFIG, "callbacks": [contador]},
            )
            respuesta = resultado["messages"][-1].content
        except Exception as e:  # noqa: BLE001
            # El detalle técnico, aparte; en el chat, un mensaje amigable.
            print(f"\n[error] {type(e).__name__}: {e}")
            respuesta = mensaje_para_el_usuario(e)
        print(f"\nSan Cris: {respuesta}")

        consumo = costos.consumo_de(contador.usage_metadata)
        total = total + consumo
        print(f"\n  [Este turno: {costos.formatear(consumo)}]")
        print(f"  [Conversación: {costos.formatear(total)}]")
        respaldo = sorted(
            m for m in contador.usage_metadata if m != llm.modelo_principal()
        )
        if respaldo:
            print(f"  [Respondido con el modelo de respaldo: {', '.join(respaldo)}]")


if __name__ == "__main__":
    main()
