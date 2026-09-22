"""Interfaz web del agente San Cris usando Streamlit.

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from config import GROQ_API_KEY, LLM_MODEL

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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("San Cris")
    st.caption("Asistente virtual de San Cristobal")
    st.divider()
    st.markdown(f"**Modelo:** `{LLM_MODEL}`")
    st.divider()
    st.markdown("**Probá con estos ejemplos:**")
    ejemplos = [
        "Hola, ¿quién sos?",
        "¿Qué plan tengo para la patente AB123CD?",
        "Choqué contra un poste con la AF987XX, ¿me cubre?",
        "Me robaron el auto AB123CD, ¿me cubre?",
    ]
    for ejemplo in ejemplos:
        st.markdown(f"- {ejemplo}")
    st.divider()
    if st.button("Reiniciar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
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

# Entrada del usuario.
if prompt := st.chat_input("Escribí tu consulta sobre tu póliza..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando póliza y cláusulas..."):
            try:
                resultado = agente.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                respuesta = resultado["messages"][-1].content
            except Exception as e:  # noqa: BLE001
                respuesta = f"Ocurrió un error: {e}"

        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
