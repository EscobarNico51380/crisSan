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

import math
import re
from datetime import date
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import database
from llm import CupoAgotado, describir_foto, segundos_de_espera
from llm import invocar as invocar_llm
from config import (
    HISTORIAL_CLASIFICADOR,
    HISTORIAL_CLASIFICADOR_CHARS,
    HISTORIAL_MENSAJES_A_CONSERVAR,
    HISTORIAL_RESUMEN_CHARS,
    HISTORIAL_UMBRAL_RESUMEN,
    PLAZO_DENUNCIA_DIAS,
)
from rag import buscar_exclusiones
from tools import (
    consultar_clausulas_poliza,
    consultar_datos_poliza,
    consultar_polizas_por_dni,
    consultar_siniestros_previos,
    registrar_siniestro,
)

# ---------------------------------------------------------------------------
# Modelo de lenguaje
# ---------------------------------------------------------------------------
# Los modelos (principal y de respaldo) se crean en ``llm.py``. Todas las
# llamadas pasan por ``invocar_llm``, que usa el primero que tenga cupo.

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


def mensaje_para_el_usuario(error: Exception) -> str:
    """Traduce un error del sistema a un mensaje apto para el chat.

    Antes el chat mostraba el error crudo de la API, con el número de
    organización de Groq incluido. El detalle técnico lo sigue viendo quien
    corre la app, en la terminal (lo imprimen ``app.py`` y ``main.py``).
    """
    texto = str(error)
    es_limite = isinstance(error, CupoAgotado) or any(
        s in texto
        for s in ("rate_limit_exceeded", "Rate limit", "rate limit", "429")
    )
    if not es_limite:
        return (
            "No pudimos procesar tu consulta. "
            "Probá de nuevo en unos minutos."
        )

    if isinstance(error, CupoAgotado):
        espera = error.segundos
    else:
        espera = segundos_de_espera(texto)
    if espera is None:
        cuando = "en unos minutos"
    else:
        minutos = max(1, math.ceil(espera / 60))
        if minutos >= 60:
            horas, resto = divmod(minutos, 60)
            cuando = f"en {horas} hora{'s' if horas > 1 else ''}"
            if resto:
                cuando += f" y {resto} minuto{'s' if resto > 1 else ''}"
        else:
            cuando = f"en {minutos} minuto{'s' if minutos > 1 else ''}"
    return (
        "El sistema no puede responder consultas en este momento. "
        f"Probá de nuevo {cuando}."
    )


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
    reclamo_resultado: str  # ficha del reclamo consultado
    reclamo_ya_denunciado: str  # reclamo existente del mismo hecho, si lo hay
    foto: str  # foto adjunta al mensaje, en base64 (solo durante el turno)
    descripcion_foto: str  # lo que describió el modelo de visión
    patente_foto: str  # patente que se lee en la foto, si se ve
    aviso_foto: str  # aviso si la foto no se pudo analizar


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


class ComparacionFoto(BaseModel):
    """¿La foto del cliente muestra lo mismo que cuenta su relato?

    Es una llamada aparte, corta y enfocada. Como un campo más del dictamen, el
    modelo chico la salteaba: en medio del prompt largo (póliza, cláusulas,
    precedentes, reglas) la comparación quedaba diluida. Aislada, los dos
    modelos la resuelven bien.
    """

    coincide: Literal["si", "no", "no aplica"] = Field(
        description=(
            "'no' si la foto muestra claramente otro tipo de daño u otra parte del "
            "auto que la que cuenta el relato (por ejemplo: relato de granizo y "
            "foto de un choque frontal). 'si' si es compatible (el granizo puede "
            "romper vidrios, por ejemplo). 'no aplica' si la foto no deja ver el "
            "daño o si el relato no dice qué pasó."
        )
    )
    diferencia: str = Field(
        default="",
        description=(
            "Si no coincide: qué cuenta el relato y qué muestra la foto, en una "
            "oración. Vacío en otro caso."
        ),
    )


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
    "Si el cliente adjuntó la FOTO de un daño y pregunta si está cubierto o "
    "cuenta qué pasó, es 'siniestro': el hecho ocurrió.\n"
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
    "4. Si se registró un siniestro, mencioná su número de reclamo.\n"
    "5. Si informás sobre un reclamo, dá sus datos completos: número, vehículo "
    "y patente, qué pasó y cuándo, el dictamen con su motivo y el estado. No "
    "respondas solo con el estado."
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

# Instrucciones para el modelo de visión. Se probaron tres redacciones con
# las fotos de ejemplo: esta fue la única que no inventó nada (con otras,
# confundía granizo con nieve o describía un vuelco que no existía). Es muy
# sensible a la redacción: ver DOCUMENTACION.md antes de cambiarla.
PROMPT_FOTO = (
    "Sos un perito que mira la foto que mandó un cliente de una aseguradora de "
    "autos. Describí SOLO lo que se ve, sin opinar sobre la cobertura.\n"
    "Reglas: afirmá solo lo que se vea con claridad; si algo no se distingue, "
    "escribí 'no se distingue'. No supongas cómo ocurrió el hecho. Si en la foto "
    "hay otros autos u objetos sueltos, no los confundas con el auto dañado.\n"
    "Respondé exactamente con este formato:\n"
    "PATENTE: <la patente si se lee con claridad, o NO SE VE>\n"
    "VEHICULO: <tipo y color, o NO SE VE UN VEHICULO>\n"
    "PARTE DAÑADA: <qué parte del auto>\n"
    "TIPO DE DAÑO: <abolladura, rotura de vidrio, granizo, choque, quemadura, faltante u otro>\n"
    "MAGNITUD: <leve, moderado o grave; parcial o posible pérdida total>\n"
    "DETALLE: <una o dos oraciones con lo que se observa, incluido el contexto si ayuda>"
)

PROMPT_COMPARAR_FOTO = (
    "Sos perito de seguros. Compará la foto del cliente con su relato: ¿muestran "
    "el mismo tipo de daño? Tené en cuenta que un mismo hecho puede dejar varios "
    "daños (el granizo abolla y también rompe vidrios)."
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
    "las salvedades de su encabezado.\n"
    "6. Si hay FOTO DEL CLIENTE: usala como evidencia del daño; si no deja ver "
    "bien el daño, basate en el relato. La descripción es automática y puede "
    "tener errores: no rechaces solo por ella.\n\n"
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


def _entrada_clasificador(
    mensajes: list[AnyMessage], resumen: str = "", descripcion_foto: str = ""
) -> str:
    """Arma lo que ve el clasificador: el mensaje actual y, si hay, los previos.

    Con el historial el clasificador entiende las preguntas de seguimiento y
    puede reescribirlas completas. Se aprovecha la llamada al LLM que ya se
    hacía para clasificar, así que no suma llamadas. En el primer mensaje de
    la conversación no hay historial y la entrada queda igual que antes.
    """
    actual = mensajes[-1].content
    previos = mensajes[:-1][-HISTORIAL_CLASIFICADOR:]
    if descripcion_foto:
        actual += (
            "\n[El cliente adjuntó una foto. Descripción automática:\n"
            f"{descripcion_foto}]"
        )
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
    try:
        resultado: AnalisisEntrada = invocar_llm(
            lambda m: m.with_structured_output(AnalisisEntrada, method="json_schema"),
            [
                SystemMessage(content=PROMPT_CLASIFICAR),
                HumanMessage(
                    content=_entrada_clasificador(
                        state["messages"],
                        state.get("resumen", ""),
                        state.get("descripcion_foto", ""),
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
        "reclamo_ya_denunciado": "",
    }


def _patente_de_la_foto(descripcion: str) -> str:
    """Lee la línea "PATENTE:" de la descripción; "" si no se ve."""
    for linea in descripcion.splitlines():
        if linea.strip().upper().startswith("PATENTE:"):
            return extraer_patente(linea.split(":", 1)[1]) or ""
    return ""


def nodo_analizar_foto(state: EstadoAgente) -> dict:
    """Si el mensaje trae una foto, el modelo de visión la describe.

    Corre antes de clasificar, así un "¿me cubre esto?" con una foto se
    entiende aunque el texto no diga qué pasó. La foto no se guarda: se borra
    del estado al terminar este paso, y con el reclamo queda solo la
    descripción. Sin foto no hace nada (ni llama al modelo).
    """
    vacio = {"foto": "", "descripcion_foto": "", "patente_foto": "", "aviso_foto": ""}
    foto = state.get("foto") or ""
    if not foto:
        return vacio
    try:
        descripcion = describir_foto(foto, PROMPT_FOTO).strip()
    except Exception as e:  # noqa: BLE001
        print(f"[foto] no se pudo analizar: {type(e).__name__}: {e}")
        return {
            **vacio,
            "aviso_foto": (
                "No pude analizar la foto que mandaste, así que esta respuesta se "
                "basa solo en tu relato."
            ),
        }
    return {
        **vacio,
        "descripcion_foto": descripcion,
        "patente_foto": _patente_de_la_foto(descripcion),
    }


def _con_aviso_de_foto(state: EstadoAgente, texto: str) -> str:
    """Antepone el aviso de foto no analizada. Lo agrega el código: el modelo
    chico ignoraba la instrucción de avisarlo."""
    aviso = state.get("aviso_foto")
    return f"⚠️ {aviso}\n\n{texto}" if aviso else texto


def nodo_pedir_patente(state: EstadoAgente) -> dict:
    texto = (
        "Para poder ayudarte necesito la patente del vehículo asegurado "
        "(por ejemplo, AB123CD). ¿Me la pasás?"
    )
    return {"messages": [AIMessage(content=_con_aviso_de_foto(state, texto))]}


def nodo_listar_polizas(state: EstadoAgente) -> dict:
    dni = state["dni"]
    resultado = consultar_polizas_por_dni.invoke({"dni": dni})
    mensaje = (
        f"Encontré estas pólizas asociadas a tu DNI:\n\n{resultado}\n\n"
        f"¿Sobre cuál patente querés que consultemos?"
    )
    return {"messages": [AIMessage(content=mensaje)]}


def nodo_consultar_reclamo(state: EstadoAgente) -> dict:
    """Responde con la ficha del reclamo armada por el código, sin el LLM.

    Antes la ficha pasaba por el LLM para redactar la respuesta, y el modelo
    podía resumirla hasta dejar solo el estado ("Iniciado."). Es una consulta
    de datos: se muestra tal cual, siempre completa, y la respuesta no gasta
    tokens (solo la clasificación del mensaje).
    """
    numero = state["reclamo"]
    datos = database.obtener_reclamo(numero)
    if datos:
        texto = database.formatear_reclamo(datos, markdown=True)
        if state.get("reclamo_ya_denunciado"):
            texto = "Ese hecho ya está denunciado. Este es tu reclamo:\n\n" + texto
    else:
        texto = (
            f"No encontré ningún reclamo con el número #{numero}. "
            "¿Podés revisar el número?"
        )
    cambios = {"reclamo_resultado": texto, "messages": [AIMessage(content=texto)]}
    if datos:
        # Así el agente sabe de qué auto se habla en las preguntas siguientes,
        # aunque la conversación haya empezado por el número de reclamo.
        cambios["patente"] = datos["patente"]
    return cambios


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


def _fecha_valida(texto: str, hoy: date) -> date:
    """Convierte la fecha que extrajo el LLM; si no es válida o es futura, hoy."""
    try:
        fecha = date.fromisoformat((texto or "").strip())
    except ValueError:
        return hoy
    return hoy if fecha > hoy else fecha


def _ddmmaaaa(fecha: date) -> str:
    return fecha.strftime("%d/%m/%Y")


def motivo_de_rechazo_administrativo(
    poliza: dict | None, fecha_siniestro: date, hoy: date | None = None
) -> str | None:
    """Controla, sin el LLM, la vigencia, el plazo de denuncia y el pago.

    Devuelve el motivo del rechazo, o ``None`` si la póliza está en regla.
    Son comparaciones de fechas y de un estado guardado en la base: no hay
    nada que interpretar, así que no se le delegan al modelo. Antes, una póliza
    vencida salía cubierta: el modelo tenía las dos fechas a la vista y no las
    comparaba.
    """
    if not poliza:
        return None
    inicio = date.fromisoformat(poliza["fecha_inicio"])
    vencimiento = date.fromisoformat(poliza["fecha_vencimiento"])
    if fecha_siniestro > vencimiento:
        return (
            f"La póliza {poliza['numero_poliza']} no estaba vigente en la fecha "
            f"del siniestro: venció el {_ddmmaaaa(vencimiento)} y el siniestro "
            f"es del {_ddmmaaaa(fecha_siniestro)}."
        )
    if fecha_siniestro < inicio:
        return (
            f"La póliza {poliza['numero_poliza']} todavía no estaba vigente en la "
            f"fecha del siniestro: empezó el {_ddmmaaaa(inicio)} y el siniestro "
            f"es del {_ddmmaaaa(fecha_siniestro)}."
        )
    # Plazo de denuncia: el que llega acá es un hecho que no estaba
    # denunciado (los ya denunciados se detectan antes), así que la denuncia es
    # hoy. Si pasaron más de 72 horas, llegó tarde.
    hoy = hoy or date.today()
    if (hoy - fecha_siniestro).days > PLAZO_DENUNCIA_DIAS:
        return (
            f"La denuncia llegó fuera de plazo: el hecho ocurrió el "
            f"{_ddmmaaaa(fecha_siniestro)} y se denuncia el {_ddmmaaaa(hoy)}. "
            f"El manual pide denunciar el siniestro dentro de las 72 horas de "
            f"ocurrido el hecho."
        )
    if poliza["estado_pago"].strip().lower() != "al dia":
        return (
            f"La póliza {poliza['numero_poliza']} tiene el pago pendiente "
            f"(estado: {poliza['estado_pago']}). Sin el pago al día no hay cobertura."
        )
    return None


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
    seccion_foto = ""
    if state.get("descripcion_foto"):
        seccion_foto = (
            "--- FOTO DEL CLIENTE (descripción automática; puede tener errores) ---\n"
            f"{state['descripcion_foto']}\n\n"
        )
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
        f"{seccion_foto}"
        f"--- RESOLUCIONES PREVIAS (precedentes) ---\n{precedentes}\n\n"
        f"--- RELATO DEL USUARIO ---\n{relato}"
    )
    resultado: DictamenSiniestro = invocar_llm(
        lambda m: m.with_structured_output(DictamenSiniestro, method="json_schema"),
        [SystemMessage(content=PROMPT_DICTAMEN), HumanMessage(content=contexto)],
    )

    # Vigencia y pago: los controla el código, no el LLM. El modelo sólo aporta
    # la fecha que entendió del relato ("ayer", "el lunes"); si no es válida o
    # es futura, se toma la de hoy. Si la póliza no estaba en regla, el
    # dictamen es "Rechazado" con un fundamento escrito por el sistema, sin
    # importar lo que haya dicho el modelo.
    hoy = date.today()
    fecha_siniestro = _fecha_valida(resultado.fecha_ocurrencia, hoy)

    # ¿Ese hecho ya estaba denunciado? "Me robaron el auto el 10 de mayo" no
    # implica que la denuncia se haga ahora: el cliente puede estar hablando de
    # un reclamo que ya hizo. Lo decide la base, no la redacción: si hay un
    # reclamo de ese auto, con esa fecha y del mismo tipo de hecho, se le
    # muestra y no se registra otro.
    existente = database.buscar_reclamo_existente(
        state.get("patente", ""), fecha_siniestro.isoformat(), resultado.tipo_siniestro
    )
    if existente:
        return {
            "dictamen": {},
            "reclamo": str(existente),
            "reclamo_ya_denunciado": str(existente),
        }

    # Comparación foto-relato: una llamada aparte, corta y enfocada.
    comparacion = None
    if state.get("descripcion_foto"):
        try:
            comparacion = invocar_llm(
                lambda m: m.with_structured_output(ComparacionFoto, method="json_schema"),
                [
                    SystemMessage(content=PROMPT_COMPARAR_FOTO),
                    HumanMessage(
                        content=(
                            "--- FOTO DEL CLIENTE (descripción automática) ---\n"
                            f"{state['descripcion_foto']}\n\n"
                            f"--- RELATO DEL CLIENTE ---\n{relato}"
                        )
                    ),
                ],
            )
        except Exception as e:  # noqa: BLE001
            # Si falla, el dictamen sigue sin esta verificación.
            print(f"[foto] no se pudo comparar con el relato: {type(e).__name__}")

    determinacion = resultado.determinacion
    fundamento = resultado.fundamento
    motivo = motivo_de_rechazo_administrativo(
        database.obtener_vigencia(state.get("patente", "")), fecha_siniestro, hoy
    )
    if motivo:
        determinacion, fundamento = "Rechazado", motivo
    elif comparacion and comparacion.coincide == "no":
        # El modelo juzga si la foto y el relato cuentan lo mismo (es
        # interpretación); la consecuencia la decide el código.
        determinacion = "Requiere peritaje"
        fundamento = (
            "La foto no coincide con el relato: "
            + (comparacion.diferencia or "muestra otro tipo de daño.")
            + " Hace falta un peritaje para determinar qué pasó."
        )
    elif state.get("patente_foto") and state["patente_foto"] != state.get("patente"):
        # La patente que se lee en la foto no es la del auto asegurado: lo
        # compara el código. No prueba fraude, pero hay que verificarlo.
        determinacion = "Requiere peritaje"
        fundamento = (
            f"La patente que se ve en la foto ({state['patente_foto']}) no coincide "
            f"con la del vehículo asegurado ({state.get('patente')}). Hace falta "
            "verificar que la foto corresponda a este auto."
        )

    return {
        "dictamen": {
            "determinacion": determinacion,
            "fundamento": fundamento,
            "tipo_siniestro": resultado.tipo_siniestro,
            "fecha_ocurrencia": fecha_siniestro.isoformat(),
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
            "fundamento": dictamen["fundamento"],
            "descripcion_foto": state.get("descripcion_foto", ""),
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
    if state.get("descripcion_foto"):
        contexto.append(
            f"[Foto del cliente, descripción automática]\n{state['descripcion_foto']}"
        )
    if state.get("dictamen"):
        d = state["dictamen"]
        contexto.append(
            f"[Dictamen]\nDeterminación: {d['determinacion']}\n"
            f"Fundamento: {d['fundamento']}"
        )
    if state.get("siniestro_registrado"):
        contexto.append(f"[Registro]\n{state['siniestro_registrado']}")

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

    respuesta = invocar_llm(lambda m: m, mensajes)
    texto = extraer_texto(respuesta.content).strip()
    if not texto:
        # Fallback defensivo: el modelo puede devolver vacío por flakiness o
        # rate limits. Nunca devolvemos una respuesta vacía al usuario.
        texto = (
            "Disculpá, no pude armar una respuesta en este momento. "
            "¿Podrías reformular tu consulta?"
        )
    return {"messages": [AIMessage(content=_con_aviso_de_foto(state, texto))]}


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
        respuesta = invocar_llm(
            lambda m: m,
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
    # Para el reclamo se mira también la pregunta reescrita por el clasificador:
    # en "¿y por qué me lo aprobaron?" la palabra "reclamo" solo aparece ahí.
    reescrita = (state.get("consulta") or "").lower()
    menciona_reclamo = any(
        palabra in texto
        for palabra in ("reclamo", "claim")
        for texto in (mensaje, reescrita)
    )
    if menciona_reclamo and state.get("reclamo"):
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

    grafo.add_node("analizar_foto", nodo_analizar_foto)
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

    # Todo turno empieza por la foto (sin foto, ese paso no hace nada).
    grafo.add_edge(START, "analizar_foto")
    grafo.add_edge("analizar_foto", "clasificar")
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
    # La ficha del reclamo ya es la respuesta: no pasa por el LLM.
    grafo.add_edge("consultar_reclamo", "compactar")
    grafo.add_edge("consultar_poliza", "consultar_clausulas")
    grafo.add_conditional_edges(
        "consultar_clausulas",
        rutear_desde_clausulas,
        {"emitir_dictamen": "emitir_dictamen", "responder": "responder"},
    )
    # Si el hecho ya estaba denunciado, se muestra ese reclamo en lugar de
    # registrar un duplicado.
    grafo.add_conditional_edges(
        "emitir_dictamen",
        lambda s: "consultar_reclamo" if s.get("reclamo_ya_denunciado") else "registrar",
        {"consultar_reclamo": "consultar_reclamo", "registrar": "registrar"},
    )
    grafo.add_edge("registrar", "responder")
    grafo.add_edge("responder", "compactar")
    grafo.add_edge("compactar", END)

    return grafo.compile(checkpointer=MemorySaver())


agente = construir_agente()
