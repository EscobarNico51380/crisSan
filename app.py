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
import llm
from config import FOTO_TAMANO_MAXIMO_MB, GROQ_API_KEY, ROOT

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
    (
        "Me cayó granizo sobre la AC456BD",
        "Con foto: el modelo de visión describe el daño y coincide con el relato.",
        "ejemplos/fotos/granizo.jpg",
    ),
    (
        "Me cayó granizo sobre la AB123CD",
        "Con la foto de un choque frontal: no coincide con el relato de granizo, "
        "así que pide peritaje.",
        "ejemplos/fotos/choque_frontal.jpg",
    ),
]

# Nombres legibles para el selector de modelo de la barra lateral.
DESCRIPCION_MODELOS = {
    "openai/gpt-oss-120b": "gpt-oss-120b · el más preciso (recomendado)",
    "openai/gpt-oss-20b": "gpt-oss-20b · más rápido y más barato",
}

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


def render_evidencia(evidencia: list[dict]) -> None:
    """Muestra la traza observable del turno.

    Dos pestañas: el razonamiento (decisiones y dictamen) y las herramientas
    efectivamente llamadas con sus argumentos y resultados. No expone el
    razonamiento interno del modelo — solo lo verificable.
    """
    if not evidencia:
        return
    with st.expander("🔍 Cómo llegué a esto"):
        tab_raz, tab_tools, tab_fuentes = st.tabs(
            ["🧠 Razonamiento", "🔧 Herramientas", "📄 Fuentes"]
        )

        with tab_raz:
            decisiones = [e for e in evidencia if e.get("tipo") == "decision"]
            dictamenes = [e for e in evidencia if e.get("tipo") == "dictamen"]
            derivaciones = [e for e in evidencia if e.get("tipo") == "derivacion"]
            if decisiones:
                st.markdown("**El modelo decidió llamar:**")
                for e in decisiones:
                    st.markdown(
                        f"- `{e.get('herramienta', '?')}` "
                        f"con `{e.get('argumentos', {})}`"
                    )
            if dictamenes:
                st.markdown("**Dictamen de cobertura:**")
                for e in dictamenes:
                    st.markdown(
                        f"- **{e.get('determinacion', '?')}** — "
                        f"{e.get('fundamento', '')}"
                    )
                    if e.get("control"):
                        # El código corrigió al modelo: se muestran las dos cosas.
                        st.markdown(
                            f"  - El modelo había dicho **{e['determinacion_modelo']}** "
                            f"({e.get('fundamento_modelo', '')}). Lo cambió un "
                            f"control del sistema: {e['control']}."
                        )
            if derivaciones:
                st.markdown("**Derivación a un asesor:**")
                for e in derivaciones:
                    st.markdown(f"- {e.get('motivo', '')}")
            if not decisiones and not dictamenes and not derivaciones:
                st.caption("Sin decisiones de herramientas en este turno.")

        with tab_tools:
            llamadas = [
                e for e in evidencia if e.get("tipo") in ("tool", "observacion")
            ]
            if llamadas:
                for e in llamadas:
                    with st.container(border=True):
                        st.markdown(f"**`{e.get('herramienta', '?')}`**")
                        if e.get("argumentos"):
                            st.caption(f"Argumentos: {e['argumentos']}")
                        st.code(str(e.get("resultado", ""))[:1500], language="text")
            else:
                st.caption("No se llamaron herramientas en este turno.")

        with tab_fuentes:
            # Chunks recuperados del manual (RAG). Aplica tanto al camino ReAct
            # (observacion) como al determinístico (tool). Cada chunk muestra su
            # sección, página y similitud coseno.
            fuentes = [
                e
                for e in evidencia
                if e.get("herramienta") == "consultar_clausulas_poliza"
                and e.get("resultado")
            ]
            if fuentes:
                for e in fuentes:
                    st.code(str(e.get("resultado", ""))[:3000], language="text")
            else:
                st.caption("No se consultó el manual en este turno.")


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
    elegido = st.selectbox(
        "Modelo de lenguaje",
        llm.MODELOS,
        index=llm.MODELOS.index(llm.modelo_principal()),
        format_func=lambda m: DESCRIPCION_MODELOS.get(m, m),
        help="Si el modelo elegido agota su cupo diario, se usa el otro automáticamente.",
    )
    llm.elegir_modelo_principal(elegido)
    respaldo = ", ".join(m for m in llm.MODELOS if m != elegido)
    st.caption(f"Respaldo si se agota el cupo diario: `{respaldo}`")
    st.divider()
    st.markdown("**Costo estimado de la conversación**")
    # Se completa al final del script, para que incluya el turno recién hecho.
    costo_total = st.empty()
    st.caption(costos.ACLARACION)
    st.divider()
    st.markdown("**Recorrido sugerido**")
    st.caption(
        "Tocá cada paso, en orden, para enviar esa consulta. También podés "
        "escribir la tuya en cualquier momento, con una foto si querés (clip del "
        "chat). Los pasos con 📷 mandan una foto de ejemplo. El recorrido completo "
        "usa unos 80.000 tokens."
    )
    for n, (pregunta, que_muestra, *foto) in enumerate(RECORRIDO, 1):
        etiqueta = f"{n}. {pregunta}" + (" 📷" if foto else "")
        if st.button(etiqueta, key=f"recorrido-{n}", use_container_width=True):
            # Se envía como si se hubiera escrito (ver el chat_input de abajo),
            # con la foto de ejemplo si el paso tiene una.
            st.session_state.pendiente = (pregunta, foto[0] if foto else None)
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

# Mostramos el historial de la conversación (con su traza).
for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        if mensaje.get("foto"):
            st.image(mensaje["foto"], width=320)
        st.markdown(mensaje["content"])
        if mensaje.get("evidencia"):
            render_evidencia(mensaje["evidencia"])
        if mensaje.get("costo"):
            st.caption(mensaje["costo"])

# Entrada del usuario.
# La consulta llega escrita en el chat o desde un botón del recorrido sugerido.
# Se puede adjuntar una foto por mensaje (clip del chat).
entrada = st.chat_input(
    "Escribí tu consulta sobre tu póliza...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=FOTO_TAMANO_MAXIMO_MB,
)
pendiente = st.session_state.pop("pendiente", None)
prompt, foto = "", None
if entrada:
    prompt = entrada.text or ""
    if entrada.files:
        foto = entrada.files[0].getvalue()
elif pendiente:
    prompt, ruta_foto = pendiente
    if ruta_foto:
        foto = (ROOT / ruta_foto).read_bytes()
if foto and not prompt:
    prompt = "Te mando una foto."

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "foto": foto})
    with st.chat_message("user"):
        if foto:
            st.image(foto, width=320)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Suma los tokens de todas las llamadas al LLM de este turno.
        contador = UsageMetadataCallbackHandler()
        with st.spinner("Consultando póliza y cláusulas..."):
            try:
                resultado = agente.invoke(
                    {
                        "messages": [HumanMessage(content=prompt)],
                        "foto": llm.preparar_foto(foto) if foto else "",
                    },
                    config={
                        "configurable": {"thread_id": st.session_state.thread_id},
                        "callbacks": [contador],
                    },
                )
                respuesta = resultado["messages"][-1].content
                evidencia = resultado.get("evidencia", [])
            except Exception as e:  # noqa: BLE001
                from agent import mensaje_para_el_usuario

                # El detalle técnico va a la terminal; al chat, un mensaje amigable.
                print(f"[error] {type(e).__name__}: {e}")
                respuesta = mensaje_para_el_usuario(e)
                evidencia = []

        consumo = costos.consumo_de(contador.usage_metadata)
        st.session_state.consumo_total = st.session_state.consumo_total + consumo
        costo_turno = f"Este turno: {costos.formatear(consumo)}"
        respaldo = sorted(
            m
            for m in contador.usage_metadata
            if m in llm.MODELOS and m != llm.modelo_principal()
        )
        if respaldo:
            costo_turno += (
                f" · Respondido con el modelo de respaldo ({', '.join(respaldo)}): "
                "el principal agotó su cupo diario"
            )

        st.markdown(respuesta)
        render_evidencia(evidencia)
        st.caption(costo_turno)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta,
            "evidencia": evidencia,
            "costo": costo_turno,
        }
    )

costo_total.markdown(costos.formatear(st.session_state.consumo_total))
