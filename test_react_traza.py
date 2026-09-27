"""Prueba del subagente ReAct + traza observable sobre la arquitectura combinada.

Verifica, sobre el agente real, que:

1. En preguntas abiertas (general, DNI), el LLM **decide** qué herramienta de
   lectura usar (patrón ReAct): la traza muestra una `decision` con herramienta
   y argumentos.
2. La traza (`evidencia`) registra la `observacion` (resultado de la tool).
3. Las consultas por patente y los siniestros van por el camino
   **determinístico** y también dejan traza (`consultar_datos_poliza`,
   `consultar_clausulas_poliza`, `dictamen`, `registrar_siniestro`).
4. La herramienta de escritura NO está expuesta al subagente ReAct.

Uso:
    .venv/bin/python test_react_traza.py
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage

import database
from agent import agente
from tools import HERRAMIENTAS_LECTURA

database.inicializar_base()

# (descripción, mensaje, camino esperado)
#   react_con_tool  -> el LLM debe decidir llamar al menos una herramienta
#   deterministico  -> consulta por patente (camino explícito, sin ReAct)
#   deterministico_siniestro -> dictamen + registro por el camino explícito
CASOS = [
    ("react: pregunta general", "¿Qué es la franquicia?", "react_con_tool"),
    ("react: pólizas por DNI", "Quiero ver mis pólizas, mi DNI es 39123456", "react_con_tool"),
    ("deterministico: consulta por patente", "¿Qué plan tengo para la patente AB123CD?", "deterministico"),
    ("deterministico: siniestro", "Me cayó granizo sobre la AF987XX, ¿me cubre?", "deterministico_siniestro"),
]


def correr(mensaje: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    r = agente.invoke({"messages": [HumanMessage(content=mensaje)]}, config=config)
    return r["messages"][-1].content, r.get("evidencia", [])


def resumir_evidencia(evidencia: list[dict]) -> str:
    if not evidencia:
        return "(vacía)"
    partes = []
    for e in evidencia:
        tipo = e.get("tipo")
        herr = e.get("herramienta") or e.get("determinacion") or "-"
        partes.append(f"{tipo}:{herr}")
    return " -> ".join(partes)


def main() -> None:
    # Chequeo estructural: la tool de escritura NO debe estar en el set ReAct.
    nombres = {t.name for t in HERRAMIENTAS_LECTURA}
    assert "registrar_siniestro" not in nombres, "registrar_siniestro no debe exponerse al LLM"
    assert "consultar_datos_poliza" in nombres
    print(f"[OK] Tools ReAct ({len(HERRAMIENTAS_LECTURA)}): {sorted(nombres)}")
    print()

    errores = 0
    for idx, (desc, mensaje, camino) in enumerate(CASOS, 1):
        try:
            respuesta, evidencia = correr(mensaje, f"traza-{idx}")
            print(f"[{idx}] {desc}  ({camino})")
            print(f"    traza: {resumir_evidencia(evidencia)}")
            print(f"    R: {' '.join(respuesta.split())[:130]}")

            tipos = {e.get("tipo") for e in evidencia}
            herramientas = {e.get("herramienta") for e in evidencia}
            if camino == "react_con_tool":
                if "decision" not in tipos:
                    errores += 1
                    print("    [!] esperaba una 'decision' del LLM y no la hubo")
            elif camino == "deterministico":
                if "consultar_datos_poliza" not in herramientas:
                    errores += 1
                    print("    [!] esperaba consultar_datos_poliza en la traza")
            else:
                # El siniestro va por el camino explícito: póliza + cláusulas.
                # (Si el hecho ya estaba denunciado, el grafo muestra el reclamo
                # existente en vez de registrar un duplicado.)
                if not {"consultar_datos_poliza", "consultar_clausulas_poliza"} <= herramientas:
                    errores += 1
                    print("    [!] esperaba el camino determinístico (póliza + cláusulas)")
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"[{idx}] {desc} -> ERROR: {str(e)[:180]}")
        print()
        time.sleep(4)  # no pisar el rate limit de Groq (30 RPM)

    print(f"===== TOTAL: {len(CASOS)} | FALLOS: {errores} =====")


if __name__ == "__main__":
    main()
