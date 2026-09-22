"""Harness de tests exhaustivos del agente San Cris.

Corre una batería de casos como si fueran clientes reales y reporta, para
cada uno, la intención clasificada, los datos extraídos (patente/DNI/reclamo),
el dictamen (si aplica) y un fragmento de la respuesta.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage

import database
from agent import agente

database.inicializar_base()

# Cada caso: (descripción corta, [mensajes en orden])
CASOS = [
    # --- Charla ---
    ("charla: hola", ["Hola"]),
    ("charla: ayuda", ["Buenas tardes, ¿me podés ayudar?"]),
    ("charla: gracias", ["Gracias"]),
    ("charla: chau", ["Chau, hasta luego"]),
    ("charla: quien sos", ["¿Quién sos?"]),
    # --- Consulta por patente ---
    ("consulta: plan", ["¿Qué plan tengo para la patente AB123CD?"]),
    ("consulta: franquicia", ["¿Cuál es mi franquicia? Mi patente es AB123CD"]),
    ("consulta: pago", ["¿Está al día mi pago? patente AF987XX"]),
    ("consulta: vencimiento", ["¿Cuándo vence mi póliza AF987XX?"]),
    ("consulta: suma asegurada", ["¿Cuánto tengo asegurado en el AB123CD?"]),
    ("consulta: datos", ["Datos de la patente AB123CD"]),
    ("consulta: patente inexistente", ["Patente ZZZ999"]),
    ("consulta: que cubre", ["¿Qué cubre mi auto AF987XX?"]),
    ("consulta: gnc", ["Tengo GNC en el AF987XX, ¿hay algún problema?"]),
    ("consulta: rastreador", ["¿Tiene rastreador mi auto AB123CD?"]),
    # --- DNI ---
    ("dni: basico", ["Mi DNI es 39123456"]),
    ("dni: con puntos", ["Quiero ver mis pólizas, DNI 39.123.456"]),
    ("dni: inexistente", ["DNI 99999999"]),
    ("dni: nombre+dni", ["Soy Carlos Perez, mi documento es 39123456"]),
    ("dni: cuantas polizas", ["¿Qué pólizas tengo a mi nombre? 39123456"]),
    # --- Siniestros ---
    (
        "siniestro: choque poste (D)",
        ["Choqué contra un poste con la AB123CD, se abolló la puerta"],
    ),
    ("siniestro: robo total (C+)", ["Me robaron el auto AF987XX"]),
    ("siniestro: incendio (D)", ["Se me prendió fuego el AB123CD"]),
    ("siniestro: granizo (C+)", ["Me cayó granizo sobre la AF987XX"]),
    ("siniestro: cristal (D)", ["Me rompieron el vidrio lateral del AB123CD"]),
    (
        "siniestro: lesionados (D)",
        ["Tuve un accidente con lesionados, patente AB123CD"],
    ),
    (
        "siniestro: choque a tercero (C+)",
        ["Choqué de atrás a otro auto, patente AF987XX"],
    ),
    ("siniestro: robo parcial (D)", ["Me robaron la rueda del AB123CD"]),
    ("siniestro: inundacion (C+)", ["El AF987XX se inundó en la calle"]),
    ("siniestro: vuelco (C+)", ["Volqué con la AF987XX en la ruta"]),
    # --- General (RAG) ---
    ("general: que es franquicia", ["¿Qué es la franquicia?"]),
    ("general: tipos franquicia", ["¿Qué tipos de franquicia hay?"]),
    ("general: responsabilidad civil", ["¿Qué es responsabilidad civil?"]),
    ("general: granizo sin patente", ["¿Me cubre el granizo?"]),
    ("general: cobertura D", ["¿Qué es la cobertura D?"]),
    ("general: exclusiones", ["¿Qué cosas excluye la póliza?"]),
    ("general: franquicia todo riesgo", ["¿Cuánto es la franquicia de todo riesgo?"]),
    ("general: no pago", ["¿Qué pasa si no pago la póliza?"]),
    ("general: cleas", ["¿Qué es CLEAS?"]),
    ("general: asistencia 24", ["¿Qué es Asistencia 24?"]),
    # --- Reclamo ---
    ("reclamo: con numero", ["¿En qué estado está mi reclamo #4?"]),
    ("reclamo: numero texto", ["Quiero saber el estado del reclamo 2"]),
    ("reclamo: inexistente", ["¿Estado del reclamo 999?"]),
    # --- Edge cases / fuera de dominio ---
    ("edge: solo patente", ["AB123CD"]),
    ("edge: fuera dominio dolar", ["¿Cuánto está el dólar?"]),
    ("edge: fuera dominio futbol", ["¿Quién ganó el mundial 2026?"]),
    ("edge: sin sentido", ["asdasdasd qwerty"]),
    (
        "edge: meteorito",
        ["¿Me cubre si cae un meteorito sobre el auto? Patente AB123CD"],
    ),
    (
        "edge: pregunta multiple",
        ["Hola, ¿está al día mi pago y qué cubre mi póliza? Patente AB123CD"],
    ),
    ("edge: mayusculas", ["QUÉ PLAN TENGO PARA LA PATENTE AF987XX"]),
    ("edge: patente con espacios", ["la patente es AB 123 CD"]),
]

MULTI = [
    # --- Multi-turno: registro y luego consulta de reclamo sin número ---
    (
        "multi: registrar luego consultar reclamo",
        [
            "Me robaron el auto AB123CD",
            "¿Cómo puedo ir viendo el estado de mi reclamo?",
        ],
    ),
    (
        "multi: patente recordada para siniestro",
        [
            "¿Qué plan tengo para la patente AF987XX?",
            "Me chocaron de atrás, se rompió el paragolpes",
        ],
    ),
]


def correr(mensajes: list[str], thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    for m in mensajes:
        agente.invoke({"messages": [HumanMessage(content=m)]}, config=config)
    snap = agente.get_state(config)
    v = snap.values
    return v, snap.values.get("messages", [])[-1].content


def resumir(v, respuesta: str) -> str:
    d = v.get("dictamen") or {}
    partes = [
        f"int={v.get('intencion')}",
        f"pat={v.get('patente') or '-'}",
        f"dni={v.get('dni') or '-'}",
        f"rec={v.get('reclamo') or '-'}",
        f"dict={d.get('determinacion', '-')}",
    ]
    return " | ".join(partes)


def main():
    # Warmup del RAG para que el primer caso que lo toca no cargue el modelo
    # de embeddings + Chroma en medio del test.
    from rag import buscar_clausulas, get_vector_store

    get_vector_store()
    buscar_clausulas("warmup")

    total = 0
    errores = 0
    for idx, (desc, mensajes) in enumerate(CASOS, 1):
        try:
            v, respuesta = correr(mensajes, f"t-{idx}")
            print(f"[{idx:02d}] {desc}")
            print(f"     {resumir(v, respuesta)}")
            print(f"     R: {' '.join(respuesta.split())[:140]}")
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"[{idx:02d}] {desc} -> ERROR: {str(e)[:180]}")
        total += 1
        print()
        time.sleep(4)  # para no pisar el rate limit de Groq (30 RPM)

    offset = len(CASOS)
    for idx, (desc, mensajes) in enumerate(MULTI, 1):
        n = offset + idx
        try:
            v, respuesta = correr(mensajes, f"m-{idx}")
            print(f"[{n:02d}] {desc}")
            print(f"     {resumir(v, respuesta)}")
            print(f"     R: {' '.join(respuesta.split())[:140]}")
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"[{n:02d}] {desc} -> ERROR: {str(e)[:180]}")
        total += 1
        print()

    print(f"===== TOTAL: {total} casos | ERRORES: {errores} =====")


if __name__ == "__main__":
    main()
