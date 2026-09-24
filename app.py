"""Interfaz web del agente San Cris usando Streamlit.

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage

import costos
from config import GROQ_API_KEY, LLM_MODEL, LLM_MODELO_RESPALDO

# Recorrido sugerido: consultas pensadas para mostrar, en orden, lo que hace el
# sistema. Cada una se envía con un botón de la barra lateral. Las que dependen
# de la anterior (la memoria, el estado del reclamo) están justo después.
RECORRIDO = [
    ("Hola, ¿quién sos?", "Charla: se presenta y cuenta qué puede hacer."),
    ("¿Qué plan tengo para la patente AB123CD?", "Consulta la póliza en la base de datos."),
    ("¿Y cuándo vence?", "Memoria: entiende que se sigue hablando del mismo auto."),
    ("¿Qué cubre el granizo?", "Busca la respuesta en el manual de la póliza (RAG)."),
    (
        "Choqué contra un poste con la AB123CD, se abolló la puerta",
        "Dictamen: todo riesgo, cubierto con franquicia. Registra el reclamo.",
    ),
    ("¿En qué estado está mi reclamo?", "Consulta el reclamo recién registrado."),
    ("Me robaron el auto AC456BD", "Robo con plan C+ y sin GNC: cubierto."),
    (
        "Me robaron el auto AF987XX",
        "El mismo robo y el mismo plan, pero el auto tiene GNC: rechazado por una "
        "exclusión del manual.",
    ),
    (
        "Me robaron el auto AJ222NP",
        "Póliza vencida: la rechaza un control del sistema, sin el LLM.",
    ),
    ("Mi DNI es 30555777", "Busca por DNI y lista las pólizas de la titular."),
]

st.set_page_config(page_title="San Cris", page_icon="🚗", layout="centered")


@st.cache_resource
def cargar_agente():
    """Inicializa la base relacional y devuelve el agente.

    Se cachea con ``cache_resource`` para que el grafo (y su memoria en
    memoria) persista entre reruns de Streamlit.
    """
    import database

    database.inicializar_base()
    from agent import agente

    return agente


# ---------------------------------------------------------------------------
# Estado de sesión (persiste entre reruns)
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "consumo_total" not in st.session_state:
    st.session_state.consumo_total = costos.Consumo()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("San Cris")
    st.caption("Asistente virtual de San Cristobal")
    st.divider()
    st.markdown(f"**Modelo:** `{LLM_MODEL}`")
    st.caption(f"Respaldo si se agota el cupo diario: `{LLM_MODELO_RESPALDO}`")
    st.divider()
    st.markdown("**Costo estimado de la conversación**")
    # Se completa al final del script, para que incluya el turno recién hecho.
    costo_total = st.empty()
    st.caption(costos.ACLARACION)
    st.divider()
    st.markdown("**Recorrido sugerido**")
    st.caption(
        "Tocá cada paso, en orden, para enviar esa consulta. También podés "
        "escribir la tuya en cualquier momento. El recorrido completo usa unos "
        "60.000 tokens."
    )
    for n, (pregunta, que_muestra) in enumerate(RECORRIDO, 1):
        if st.button(f"{n}. {pregunta}", key=f"recorrido-{n}", use_container_width=True):
            # Se envía como si se hubiera escrito (ver el chat_input de abajo).
            st.session_state.pendiente = pregunta
        st.caption(que_muestra)
    st.divider()
    if st.button("Reiniciar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.consumo_total = costos.Consumo()
        st.rerun()


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------
st.title("San Cris")
st.caption("Agente conversacional de seguros — RAG + LangGraph + Groq")

if not GROQ_API_KEY:
    st.error(
        "No se encontró `GROQ_API_KEY` en el `.env`. Configuralo y reiniciá la app."
    )
    st.stop()

agente = cargar_agente()

# Mostramos el historial de la conversación.
for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        if mensaje.get("costo"):
            st.caption(mensaje["costo"])

# Entrada del usuario.
# La consulta llega escrita en el chat o desde un botón del recorrido sugerido.
pregunta_del_recorrido = st.session_state.pop("pendiente", None)
if prompt := (
    st.chat_input("Escribí tu consulta sobre tu póliza...") or pregunta_del_recorrido
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Suma los tokens de todas las llamadas al LLM de este turno.
        contador = UsageMetadataCallbackHandler()
        with st.spinner("Consultando póliza y cláusulas..."):
            try:
                resultado = agente.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={
                        "configurable": {"thread_id": st.session_state.thread_id},
                        "callbacks": [contador],
                    },
                )
                respuesta = resultado["messages"][-1].content
            except Exception as e:  # noqa: BLE001
                from agent import mensaje_para_el_usuario

                # El detalle técnico va a la terminal; al chat, un mensaje amigable.
                print(f"[error] {type(e).__name__}: {e}")
                respuesta = mensaje_para_el_usuario(e)

        consumo = costos.consumo_de(contador.usage_metadata)
        st.session_state.consumo_total = st.session_state.consumo_total + consumo
        costo_turno = f"Este turno: {costos.formatear(consumo)}"
        respaldo = sorted(m for m in contador.usage_metadata if m != LLM_MODEL)
        if respaldo:
            costo_turno += (
                f" · Respondido con el modelo de respaldo ({', '.join(respaldo)}): "
                "el principal agotó su cupo diario"
            )

        st.markdown(respuesta)
        st.caption(costo_turno)

    st.session_state.messages.append(
        {"role": "assistant", "content": respuesta, "costo": costo_turno}
    )

costo_total.markdown(costos.formatear(st.session_state.consumo_total))
