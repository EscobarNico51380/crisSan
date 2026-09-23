"""Agente conversacional basado en un grafo de estado de LangGraph.

El flujo está controlado de forma EXPLÍCITA (no es un ReAct a ciegas):

    START
      -> clasificar (LLM: intención + patente)
            |-- "charla"  -> responder -> END
            |-- "general" -> consultar_clausulas -> responder -> END
            |-- "consulta"/"siniestro" sin patente -> pedir_patente -> END
            |-- "consulta"  -> consultar_poliza -> consultar_clausulas
                                   -> responder -> END
            |-- "siniestro" -> consultar_poliza -> consultar_clausulas
                                   -> emitir_dictamen -> registrar
                                   -> responder -> END

La memoria conversacional se logra con el checkpointer (MemorySaver) y un
thread_id por sesión: el historial vive en el estado ``messages``.
"""

from __future__ import annotations

import re
import time
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import database
from config import (
    HISTORIAL_CLASIFICADOR,
    HISTORIAL_CLASIFICADOR_CHARS,
    HISTORIAL_MENSAJES_A_CONSERVAR,
    HISTORIAL_RESUMEN_CHARS,
    HISTORIAL_UMBRAL_RESUMEN,
    LLM_MODEL,
)
from rag import buscar_exclusiones
from tools import (
    consultar_clausulas_poliza,
    consultar_datos_poliza,
    consultar_estado_reclamo,
    consultar_polizas_por_dni,
    consultar_siniestros_previos,
    registrar_siniestro,
)

# ---------------------------------------------------------------------------
# Modelo de lenguaje
# ---------------------------------------------------------------------------
llm = ChatGroq(model=LLM_MODEL, temperature=0)

# Patente argentina: formato Mercosur (AB123CD) o formato viejo (ABC123).
# Toleramos separadores opcionales: "AB 123 CD", "AB-123-CD" o "AB123CD".
PATRON_PATENTE = re.compile(
    r"\b[A-Za-z]{2}[\s-]?\d{3}[\s-]?[A-Za-z]{2}\b|\b[A-Za-z]{3}[\s-]?\d{3}\b"
)

# DNI argentino: 8 dígitos, con o sin puntos (39.123.456 o 39123456).
PATRON_DNI = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}\b")

# Número de reclamo: "reclamo #8", "reclamo 8", "claim 8", "número 8".
PATRON_RECLAMO = re.compile(r"(?:reclamo|claim|n[úu]mero)\s*#?\s*(\d+)", re.IGNORECASE)


def extraer_patente(texto: str) -> str | None:
    """Extrae una patente argentina del texto de forma determinística.

    No depende del LLM, así que es robusta ante inputs ambiguos (ej. que el
    usuario escriba solo "AB123CD").
    """
    coincidencia = PATRON_PATENTE.search(texto)
    if coincidencia:
        # Normalizamos quitando espacios y guiones ("AB 123 CD" -> "AB123CD").
        return re.sub(r"[^A-Za-z0-9]", "", coincidencia.group(0)).upper()
    return None


def extraer_dni(texto: str) -> str | None:
    """Extrae un DNI argentino (8 dígitos) del texto de forma determinística."""
    coincidencia = PATRON_DNI.search(texto)
    if coincidencia:
        # Normalizamos quitando los puntos (39.123.456 -> 39123456).
        return coincidencia.group(0).replace(".", "")
    return None


def extraer_reclamo(texto: str) -> str | None:
    """Extrae un número de reclamo del texto ("reclamo #8", "reclamo 8")."""
    coincidencia = PATRON_RECLAMO.search(texto)
    if coincidencia:
        return coincidencia.group(1)
    return None


def extraer_texto(content) -> str:
    """Extrae texto plano de un contenido de mensaje.

    Algunos proveedores devuelven el contenido como una lista de bloques
    en lugar de un string. Este helper lo normaliza a str de forma defensiva
    (si ya es string, lo devuelve tal cual).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                texto = bloque.get("text") or bloque.get("content")
                if isinstance(texto, str) and texto:
                    partes.append(texto)
            else:
                texto = getattr(bloque, "text", None) or getattr(
                    bloque, "content", None
                )
                if isinstance(texto, str) and texto:
                    partes.append(texto)
        return "\n".join(partes).strip()
    return str(content)


def invocar_con_reintentos(func, *args, **kwargs):
    """Reintenta llamadas al LLM ante errores transitorios (429/503).

    Tanto la API gratuita de Gemini como la de Groq pueden responder
    temporalmente 429 (rate limit) o 503 (alta demanda). Hacemos backoff
    exponencial y reintentamos hasta ``intentos`` veces.
    """
    intentos = kwargs.pop("_intentos", 5)
    ultimo_error = None
    for i in range(intentos):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            mensaje = str(e)
            # El límite diario de tokens (TPD) no se resuelve con reintentos.
            if "tokens per day" in mensaje or "TPD" in mensaje:
                raise
            es_transitorio = any(
                token in mensaje
                for token in (
                    "503",
                    "429",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "high demand",
                    "rate limit",
                    "Rate limit",
                )
            )
            if es_transitorio and i < intentos - 1:
                time.sleep(2**i)  # 1s, 2s, 4s, 8s...
                continue
            raise
    raise ultimo_error


# ---------------------------------------------------------------------------
# Exclusiones que dependen del vehículo
# ---------------------------------------------------------------------------
# El cliente no las menciona ("me robaron el auto" no dice que tiene GNC), así
# que la búsqueda con su mensaje no las trae. Se buscan aparte, con una
# consulta armada por el sistema, sólo si el vehículo tiene la característica.
CONSULTA_EXCLUSIONES_GNC = (
    "exclusiones vehiculo adaptado para ser propulsado por gas natural comprimido GNC"
)
CLAVES_GNC = ("gnc", "gas natural comprimido")


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class EstadoAgente(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # historial reciente
    resumen: str  # resumen de los mensajes viejos (persistente entre turnos)
    intencion: str  # consulta | siniestro | general | charla
    consulta: str  # mensaje reescrito para entenderse sin el historial
    patente: str  # patente extraída (persistente entre turnos)
    dni: str  # DNI del titular (si lo menciona)
    reclamo: str  # número de reclamo (persistente entre turnos)
    datos_poliza: str  # resultado de consultar_datos_poliza
    clausulas: str  # resultado del RAG
    exclusiones: str  # exclusiones que dependen del vehículo (ej. GNC)
    dictamen: dict  # dictamen estructurado del siniestro
    siniestro_registrado: str  # resultado del registro del siniestro
    reclamo_resultado: str  # resultado de consultar_estado_reclamo


# ---------------------------------------------------------------------------
# Schemas de salida estructurada
# ---------------------------------------------------------------------------
class AnalisisEntrada(BaseModel):
    """Clasificación de la intención del usuario + datos extraídos."""

    intencion: Literal["consulta", "siniestro", "general", "charla"] = Field(
        description=(
            "consulta: pregunta sobre SU póliza (estado, plan, franquicia, "
            "vencimiento). siniestro: describe un accidente/robo/incendio/"
            "granizo y quiere saber si está cubierto. general: pregunta "
            "genérica sobre el manual/coberturas. charla: saludo o trivial."
        )
    )
    patente: Optional[str] = Field(
        default=None,
        description="Patente argentina (3 letras + 3 números, ej AB123CD) si aparece",
    )
    dni: Optional[str] = Field(
        default=None,
        description="DNI del titular (8 dígitos) si aparece",
    )
    consulta_autonoma: str = Field(
        default="",
        description=(
            "El MENSAJE A CLASIFICAR reescrito para que se entienda sin el "
            "historial: incorporá la patente, el tipo de siniestro o el tema "
            "que se venía hablando. Si ya se entiende solo, copialo igual."
        ),
    )


class DictamenSiniestro(BaseModel):
    """Determinación de cobertura + datos del siniestro para registrar."""

    determinacion: Literal[
        "Cubierto", "Rechazado", "Requiere peritaje", "Asistencia 24"
    ] = Field(description="Determinación de si el seguro cubre o no el siniestro")
    fundamento: str = Field(description="Justificación basada en póliza y cláusulas")
    tipo_siniestro: str = Field(
        description="Ej: Accidente, Robo Total, Granizo, Incendio"
    )
    fecha_ocurrencia: str = Field(
        description="Fecha en formato YYYY-MM-DD (hoy si no se aclara)"
    )
    hubo_lesionados: int = Field(description="1 si hubo lesionados, 0 si no")


# ---------------------------------------------------------------------------
# Prompts del sistema
# ---------------------------------------------------------------------------
PROMPT_CLASIFICAR = (
    "Sos el ROUTER de un sistema RAG de seguros San Cristóbal.\n"
    "Clasificá el mensaje en UNA de estas intenciones:\n"
    "- 'consulta': pregunta o da datos de SU póliza (patente o DNI).\n"
    "- 'siniestro': reporta un hecho ocurrido (choque, robo, incendio, granizo, vuelco).\n"
    "- 'general': pregunta genérica sobre el manual/coberturas, sin su póliza.\n"
    "- 'charla': saludo, agradecimiento o texto trivial.\n"
    "Regla: si menciona patente o DNI → 'consulta' (salvo que reporte un hecho → 'siniestro').\n"
    "Si hay HISTORIAL RECIENTE, usalo para entender a qué se refiere el mensaje "
    "('¿y cuándo vence?' después de hablar de una póliza es 'consulta'), pero "
    "clasificá SOLO el MENSAJE A CLASIFICAR.\n"
    "'siniestro' es sólo un hecho que OCURRIÓ. Una hipótesis ('¿y si hubiera "
    "sido granizo?', '¿me cubriría si...?') es 'consulta' si hay una póliza en "
    "juego, o 'general' si no.\n"
    "Además devolvé el mensaje reescrito para que se entienda sin el historial.\n"
    "Respondé SOLO con la estructura pedida, sin texto adicional."
)

PROMPT_RESPONDER = (
    "Sos 'San Cris', el asistente virtual de seguros San Cristóbal.\n"
    "Reglas:\n"
    "1. Respondé SOLO con la información provista (datos de la póliza y "
    "cláusulas del manual). No inventes coberturas, montos, emails, "
    "teléfonos, portales ni procedimientos.\n"
    "2. Si no tenés la información, decilo con honestidad y pedí los datos "
    "que falten (patente, DNI o número de reclamo).\n"
    "3. Español rioplatense, claro y directo. Sin muletillas iniciales "
    "('¡Claro!', '¡Por supuesto!'). Si el usuario saluda, devolvé el saludo.\n"
    "4. Si se registró un siniestro, mencioná su número de reclamo."
)

# Para los mensajes de charla (saludos, agradecimientos, temas ajenos). Sin
# esta instrucción, la charla llegaba al nodo que responde sin contexto y
# recibía la advertencia pensada para consultas sin datos: al "Hola" le
# contestaba "no dispongo de la información necesaria".
PROMPT_CHARLA = (
    "Es un mensaje de conversación: un saludo, un agradecimiento, una "
    "despedida, una pregunta sobre vos o algo ajeno a los seguros. Respondé "
    "con naturalidad y en pocas líneas.\n"
    "- Si saluda o pregunta quién sos o para qué servís: presentate como San "
    "Cris, el asistente de seguros de autos de San Cristóbal, y contá qué "
    "podés hacer: consultar su póliza (plan, franquicia, vencimiento, pago) con "
    "la patente o el DNI; responder dudas sobre las coberturas del manual; "
    "analizar un siniestro, decir si está cubierto y registrar el reclamo; y "
    "consultar el estado de un reclamo.\n"
    "- Si el tema no es de seguros (el dólar, fútbol, el clima): decí "
    "amablemente que no podés ayudar con eso y ofrecé lo que sí podés hacer.\n"
    "- Si no se entiende el mensaje, pedí que lo reformule.\n"
    "No inventes datos de pólizas, montos, teléfonos ni procedimientos."
)

PROMPT_DICTAMEN = (
    "Sos el PERITO de seguros San Cristóbal. Determiná si el siniestro está "
    "CUBIERTO o NO.\n\n"
    "COBERTURA POR PRODUCTO:\n"
    "- A (Auto Base): solo Responsabilidad Civil (daños a terceros). No cubre el propio auto.\n"
    "- B (Auto Mix): daños TOTALES (robo/hurto total, incendio total, "
    "destrucción total, vuelco, despeñamiento, inmersión). No cubre parciales.\n"
    "- C/C+/CM (Terceros Completos): RC + robo/hurto + incendio + parciales "
    "específicos (cristales, cerraduras, parabrisas, luneta, granizo). "
    "No cubre parciales por colisión/choque.\n"
    "- D (Todo Riesgo): cubre todo, incluidos parciales y totales por choque/accidente.\n\n"
    "DECISIÓN:\n"
    "1. Si el siniestro no está en las coberturas del producto → 'Rechazado' (fundamentá por qué).\n"
    "2. Considerá franquicia, exclusiones y estado de pago (impago → rechazá).\n"
    "3. Si falta un dato clave (daño total vs parcial) → 'Requiere peritaje'.\n"
    "4. Extraé tipo de siniestro, fecha (hoy si no se aclara) y si hubo lesionados.\n"
    "5. Si hay EXCLUSIONES POR CARACTERÍSTICAS DEL VEHÍCULO: cada una indica entre "
    "corchetes a qué cobertura aplica. Aplicala SOLO a esa cobertura y respetá "
    "las salvedades de su encabezado.\n\n"
    "Básate en: la póliza, las cláusulas recuperadas, las exclusiones y las "
    "resoluciones previas (referencia, no regla rígida)."
)


PROMPT_RESUMEN = (
    "Resumí la conversación entre un cliente y San Cris, el asistente de seguros "
    "San Cristóbal, para que el asistente pueda seguirla sin ver los mensajes "
    "originales.\n"
    "Conservá: patentes y DNI mencionados, siniestros denunciados con su número "
    "de reclamo y dictamen, datos de pólizas consultados y qué quiere o "
    "preguntó el cliente.\n"
    "Si hay un RESUMEN ANTERIOR, actualizalo con los mensajes nuevos: no lo "
    "descartes.\n"
    "No inventes nada que no esté en el texto. Máximo 120 palabras, en tercera "
    "persona, sin encabezados."
)


# ---------------------------------------------------------------------------
# Nodos
# ---------------------------------------------------------------------------
def ventana_historial(mensajes: list[AnyMessage], maximo: int) -> list[AnyMessage]:
    """Devuelve hasta los últimos ``maximo`` mensajes, empezando por el usuario.

    Si la ventana arrancara con una respuesta del asistente, el LLM la vería sin
    la pregunta que la originó, así que se descartan las del principio hasta
    llegar a un mensaje del usuario.
    """
    ventana = list(mensajes[-maximo:])
    while ventana and not isinstance(ventana[0], HumanMessage):
        ventana.pop(0)
    return ventana


def _transcripcion(mensajes: list[AnyMessage], tope: int) -> str:
    """Pasa mensajes a texto "Usuario: ... / Asistente: ...", recortando cada uno."""
    lineas = []
    for m in mensajes:
        quien = "Usuario" if isinstance(m, HumanMessage) else "Asistente"
        texto = " ".join(extraer_texto(m.content).split())
        if len(texto) > tope:
            texto = texto[:tope] + "…"
        lineas.append(f"{quien}: {texto}")
    return "\n".join(lineas)


def _entrada_clasificador(mensajes: list[AnyMessage], resumen: str = "") -> str:
    """Arma lo que ve el clasificador: el mensaje actual y, si hay, los previos.

    Con el historial el clasificador entiende las preguntas de seguimiento y
    puede reescribirlas completas. Se aprovecha la llamada al LLM que ya se
    hacía para clasificar, así que no suma llamadas. En el primer mensaje de
    la conversación no hay historial y la entrada queda igual que antes.
    """
    actual = mensajes[-1].content
    previos = mensajes[:-1][-HISTORIAL_CLASIFICADOR:]
    if not previos and not resumen:
        return actual
    partes = []
    if resumen:
        partes.append(f"RESUMEN DE LA CONVERSACIÓN ANTERIOR:\n{resumen}")
    if previos:
        partes.append(
            "HISTORIAL RECIENTE (para entender el mensaje, no para clasificarlo):\n"
            + _transcripcion(previos, HISTORIAL_CLASIFICADOR_CHARS)
        )
    partes.append(f"MENSAJE A CLASIFICAR:\n{actual}")
    return "\n\n".join(partes)


def nodo_clasificar(state: EstadoAgente) -> dict:
    mensaje = state["messages"][-1].content

    # La patente, el DNI y el reclamo se extraen con regex (determinístico).
    patente_regex = extraer_patente(mensaje)
    dni_regex = extraer_dni(mensaje)
    reclamo_regex = extraer_reclamo(mensaje)

    # json_schema usa el structured output nativo de Groq: fuerza el schema
    # y evita tool calls malformados (como {"consulta": ...} en vez de
    # {"intencion": ...}).
    clasificador = llm.with_structured_output(AnalisisEntrada, method="json_schema")
    try:
        resultado: AnalisisEntrada = invocar_con_reintentos(
            clasificador.invoke,
            [
                SystemMessage(content=PROMPT_CLASIFICAR),
                HumanMessage(
                    content=_entrada_clasificador(
                        state["messages"], state.get("resumen", "")
                    )
                ),
            ],
        )
        intencion = resultado.intencion
        consulta = (resultado.consulta_autonoma or "").strip() or mensaje
        patente = patente_regex or (resultado.patente or "")
        dni = dni_regex or (resultado.dni or "")
    except Exception:  # noqa: BLE001
        # Fallback determinístico ante un fallo del LLM: si hay patente en el
        # texto, asumimos consulta; si no, charla. Mejor que romper la app.
        intencion = "consulta" if patente_regex else "charla"
        consulta = mensaje
        patente = patente_regex or ""
        dni = dni_regex or ""

    # Normalizamos la patente (venga del regex o del LLM): quitamos
    # espacios/guiones ("AB 123 CD" -> "AB123CD").
    if patente:
        patente = re.sub(r"[^A-Za-z0-9]", "", patente).upper()

    # Memoria: si este mensaje no menciona una patente, conservamos la que el
    # usuario ya dio en turnos anteriores.
    if not patente:
        patente = state.get("patente", "")

    # Ídem con el DNI: si no se menciona uno nuevo, conservamos el último.
    if not dni:
        dni = state.get("dni", "")

    # Ídem con el número de reclamo: si no se menciona uno nuevo, se conserva
    # el último registrado en la conversación.
    reclamo = reclamo_regex or state.get("reclamo", "")

    # Reseteamos los campos "por turno" para que no queden datos viejos de
    # una consulta anterior contaminando la respuesta actual.
    return {
        "intencion": intencion,
        "consulta": consulta,
        "patente": patente,
        "dni": dni,
        "reclamo": reclamo,
        "datos_poliza": "",
        "clausulas": "",
        "dictamen": {},
        "siniestro_registrado": "",
        "reclamo_resultado": "",
        "exclusiones": "",
    }


def nodo_pedir_patente(state: EstadoAgente) -> dict:
    texto = (
        "Para poder ayudarte necesito la patente del vehículo asegurado "
        "(por ejemplo, AB123CD). ¿Me la pasás?"
    )
    return {"messages": [AIMessage(content=texto)]}


def nodo_listar_polizas(state: EstadoAgente) -> dict:
    dni = state["dni"]
    resultado = consultar_polizas_por_dni.invoke({"dni": dni})
    mensaje = (
        f"Encontré estas pólizas asociadas a tu DNI:\n\n{resultado}\n\n"
        f"¿Sobre cuál patente querés que consultemos?"
    )
    return {"messages": [AIMessage(content=mensaje)]}


def nodo_consultar_reclamo(state: EstadoAgente) -> dict:
    reclamo = state["reclamo"]
    resultado = consultar_estado_reclamo.invoke({"numero_reclamo": reclamo})
    return {"reclamo_resultado": resultado}


def nodo_consultar_poliza(state: EstadoAgente) -> dict:
    patente = state["patente"]
    resultado = consultar_datos_poliza.invoke({"patente": patente})
    return {"datos_poliza": resultado}


def nodo_consultar_clausulas(state: EstadoAgente) -> dict:
    # Buscamos con la versión reescrita por el clasificador: un seguimiento
    # como "¿y con granizo?" suelto recupera cláusulas al azar.
    consulta = state.get("consulta") or state["messages"][-1].content
    resultado = consultar_clausulas_poliza.invoke({"consulta": consulta})

    # Segunda búsqueda, dirigida a las exclusiones que dependen del vehículo.
    # Sin ella el dictamen cubría un robo o un incendio de un auto con GNC,
    # porque esas exclusiones nunca aparecían entre los fragmentos recuperados.
    # Sólo para consultas y siniestros: en una pregunta general la patente
    # recordada de un turno anterior no tiene por qué pesar.
    exclusiones = ""
    patente = state.get("patente", "")
    if (
        state.get("intencion") in ("consulta", "siniestro")
        and patente
        and database.vehiculo_tiene_gnc(patente)
    ):
        exclusiones = buscar_exclusiones(CONSULTA_EXCLUSIONES_GNC, CLAVES_GNC)
    return {"clausulas": resultado, "exclusiones": exclusiones}


def nodo_emitir_dictamen(state: EstadoAgente) -> dict:
    mensaje = state["messages"][-1].content
    # Acotamos los precedentes al MISMO producto que la póliza en cuestión: una
    # resolución sobre un Todo Riesgo no es comparable con una de Terceros
    # Completos. No filtramos por tipo_siniestro porque todavía no se conoce
    # (lo determina el LLM más abajo, en esta misma función).
    precedentes = consultar_siniestros_previos.invoke(
        {"limite": 5, "patente": state.get("patente", "")}
    )
    # Si el mensaje dependía del historial, sumamos la versión reescrita para
    # que el perito sepa de qué hecho se habla. En la base se guarda el
    # mensaje original (nodo_registrar).
    relato = mensaje
    consulta = state.get("consulta", "")
    if consulta and consulta.strip() != mensaje.strip():
        relato += f"\n(Reformulado con el contexto de la conversación: {consulta})"
    seccion_exclusiones = ""
    if state.get("exclusiones"):
        seccion_exclusiones = (
            "--- EXCLUSIONES POR CARACTERÍSTICAS DEL VEHÍCULO "
            "(el vehículo tiene GNC) ---\n"
            f"{state['exclusiones']}\n\n"
        )
    contexto = (
        f"--- DATOS DE LA PÓLIZA ---\n{state['datos_poliza']}\n\n"
        f"--- CLÁUSULAS DEL MANUAL ---\n{state['clausulas']}\n\n"
        f"{seccion_exclusiones}"
        f"--- RESOLUCIONES PREVIAS (precedentes) ---\n{precedentes}\n\n"
        f"--- RELATO DEL USUARIO ---\n{relato}"
    )
    dictaminador = llm.with_structured_output(DictamenSiniestro, method="json_schema")
    resultado: DictamenSiniestro = invocar_con_reintentos(
        dictaminador.invoke,
        [SystemMessage(content=PROMPT_DICTAMEN), HumanMessage(content=contexto)],
    )
    return {
        "dictamen": {
            "determinacion": resultado.determinacion,
            "fundamento": resultado.fundamento,
            "tipo_siniestro": resultado.tipo_siniestro,
            "fecha_ocurrencia": resultado.fecha_ocurrencia,
            "hubo_lesionados": resultado.hubo_lesionados,
        }
    }


def nodo_registrar(state: EstadoAgente) -> dict:
    dictamen = state["dictamen"]
    descripcion = state["messages"][-1].content  # relato original del usuario
    resultado = registrar_siniestro.invoke(
        {
            "patente": state["patente"],
            "tipo_siniestro": dictamen["tipo_siniestro"],
            "fecha_ocurrencia": dictamen["fecha_ocurrencia"],
            "hubo_lesionados": dictamen["hubo_lesionados"],
            "descripcion": descripcion,
            "dictamen": dictamen["determinacion"],
        }
    )
    # Guardamos el número de reclamo para poder consultarlo en turnos siguientes.
    m = re.search(r"#(\d+)", resultado)
    reclamo = m.group(1) if m else ""
    return {"siniestro_registrado": resultado, "reclamo": reclamo}


def nodo_responder(state: EstadoAgente) -> dict:
    contexto = []
    if state.get("datos_poliza"):
        contexto.append(f"[Datos de la póliza]\n{state['datos_poliza']}")
    if state.get("clausulas"):
        contexto.append(f"[Cláusulas del manual]\n{state['clausulas']}")
    if state.get("exclusiones"):
        contexto.append(
            f"[Exclusiones por características del vehículo]\n{state['exclusiones']}"
        )
    if state.get("dictamen"):
        d = state["dictamen"]
        contexto.append(
            f"[Dictamen]\nDeterminación: {d['determinacion']}\n"
            f"Fundamento: {d['fundamento']}"
        )
    if state.get("siniestro_registrado"):
        contexto.append(f"[Registro]\n{state['siniestro_registrado']}")
    if state.get("reclamo_resultado"):
        contexto.append(f"[Estado del reclamo]\n{state['reclamo_resultado']}")

    mensajes = [SystemMessage(content=PROMPT_RESPONDER)]
    if state.get("resumen"):
        mensajes.append(
            SystemMessage(
                content=f"RESUMEN DE LA CONVERSACIÓN HASTA AHORA:\n{state['resumen']}"
            )
        )
    if contexto:
        mensajes.append(SystemMessage(content="\n\n".join(contexto)))
    elif state.get("intencion") == "charla":
        # La charla no necesita datos: se le indica cómo conversar.
        mensajes.append(SystemMessage(content=PROMPT_CHARLA))
    else:
        # Sin contexto: prohibimos inventar información.
        mensajes.append(
            SystemMessage(
                content=(
                    "ATENCIÓN: no tenés información de contexto para responder "
                    "esta consulta. NO inventes portales, apps, emails, "
                    "teléfonos ni procedimientos. Respondé honestamente que no "
                    "podés responder eso con la información disponible y pedí "
                    "los datos que te permitan buscarlo (patente, DNI o número "
                    "de reclamo)."
                )
            )
        )
    mensajes.extend(state["messages"])

    respuesta = invocar_con_reintentos(llm.invoke, mensajes)
    texto = extraer_texto(respuesta.content).strip()
    if not texto:
        # Fallback defensivo: el modelo puede devolver vacío por flakiness o
        # rate limits. Nunca devolvemos una respuesta vacía al usuario.
        texto = (
            "Disculpá, no pude armar una respuesta en este momento. "
            "¿Podrías reformular tu consulta?"
        )
    return {"messages": [AIMessage(content=texto)]}


def nodo_compactar(state: EstadoAgente) -> dict:
    """Comprime los mensajes viejos en el resumen cuando el historial es largo.

    Corre al final de cada turno, así la conversación tiene pares completos y
    la ventana que queda empieza con una pregunta del usuario. Si no se llegó
    al umbral no hace nada (ni llama al LLM).

    Los mensajes resumidos se borran del estado con ``RemoveMessage``. La
    pantalla de Streamlit guarda su propia copia del chat, así que el usuario
    sigue viendo todo. El dictamen tampoco depende del resumen: arma su
    contexto con datos frescos de la base y del manual.
    """
    mensajes = state["messages"]
    if len(mensajes) < HISTORIAL_UMBRAL_RESUMEN:
        return {}

    conservar = ventana_historial(mensajes, HISTORIAL_MENSAJES_A_CONSERVAR)
    viejos = mensajes[: len(mensajes) - len(conservar)]
    if not viejos:
        return {}

    pedido = (
        f"RESUMEN ANTERIOR:\n{state.get('resumen') or '(todavía no hay)'}\n\n"
        f"MENSAJES NUEVOS A INCORPORAR:\n"
        f"{_transcripcion(viejos, HISTORIAL_RESUMEN_CHARS)}"
    )
    try:
        respuesta = invocar_con_reintentos(
            llm.invoke,
            [SystemMessage(content=PROMPT_RESUMEN), HumanMessage(content=pedido)],
        )
        resumen = extraer_texto(respuesta.content).strip()
    except Exception:  # noqa: BLE001
        # Si el resumen falla no borramos nada: es preferible un historial más
        # largo por un turno que perder contexto.
        return {}
    if not resumen:
        return {}

    return {
        "resumen": resumen,
        "messages": [RemoveMessage(id=m.id) for m in viejos],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def rutear_desde_clasificar(state: EstadoAgente) -> str:
    intencion = state["intencion"]
    mensaje = state["messages"][-1].content.lower()

    # Señales EXPLÍCITAS del mensaje actual (determinísticas, máxima prioridad).
    if ("reclamo" in mensaje or "claim" in mensaje) and state.get("reclamo"):
        return "consultar_reclamo"
    if extraer_dni(mensaje) and not extraer_patente(mensaje):
        return "listar_polizas"
    if extraer_patente(mensaje):
        return "consultar_poliza"

    # Intención del LLM (con valores persistidos como respaldo).
    if intencion == "charla":
        return "responder"
    if intencion == "general":
        return "consultar_clausulas"
    if state.get("patente"):
        # consulta/siniestro con patente (actual o recordada en la conversación).
        return "consultar_poliza"
    if intencion == "consulta" and state.get("dni"):
        return "listar_polizas"
    return "pedir_patente"


def rutear_desde_consultar_poliza(state: EstadoAgente) -> str:
    return "consultar_clausulas"


def rutear_desde_clausulas(state: EstadoAgente) -> str:
    return "emitir_dictamen" if state["intencion"] == "siniestro" else "responder"


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def construir_agente():
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("clasificar", nodo_clasificar)
    grafo.add_node("pedir_patente", nodo_pedir_patente)
    grafo.add_node("listar_polizas", nodo_listar_polizas)
    grafo.add_node("consultar_reclamo", nodo_consultar_reclamo)
    grafo.add_node("consultar_poliza", nodo_consultar_poliza)
    grafo.add_node("consultar_clausulas", nodo_consultar_clausulas)
    grafo.add_node("emitir_dictamen", nodo_emitir_dictamen)
    grafo.add_node("registrar", nodo_registrar)
    grafo.add_node("responder", nodo_responder)
    grafo.add_node("compactar", nodo_compactar)

    grafo.add_edge(START, "clasificar")
    grafo.add_conditional_edges(
        "clasificar",
        rutear_desde_clasificar,
        {
            "responder": "responder",
            "consultar_clausulas": "consultar_clausulas",
            "pedir_patente": "pedir_patente",
            "listar_polizas": "listar_polizas",
            "consultar_reclamo": "consultar_reclamo",
            "consultar_poliza": "consultar_poliza",
        },
    )
    # Todos los caminos terminan pasando por "compactar", que sólo actúa
    # cuando el historial llegó al umbral.
    grafo.add_edge("pedir_patente", "compactar")
    grafo.add_edge("listar_polizas", "compactar")
    grafo.add_edge("consultar_reclamo", "responder")
    grafo.add_edge("consultar_poliza", "consultar_clausulas")
    grafo.add_conditional_edges(
        "consultar_clausulas",
        rutear_desde_clausulas,
        {"emitir_dictamen": "emitir_dictamen", "responder": "responder"},
    )
    grafo.add_edge("emitir_dictamen", "registrar")
    grafo.add_edge("registrar", "responder")
    grafo.add_edge("responder", "compactar")
    grafo.add_edge("compactar", END)

    return grafo.compile(checkpointer=MemorySaver())


agente = construir_agente()
