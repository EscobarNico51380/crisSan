"""Acceso al modelo de lenguaje: modelo principal, modelo de respaldo y disyuntor.

El plan gratuito de Groq limita los tokens diarios **por modelo**. Cuando el
modelo principal agota su cupo, las consultas pasan al de respaldo, que tiene
el suyo propio.

**Disyuntor.** Cuando un modelo responde "cupo diario agotado, probá de nuevo en
X", se anota hasta cuándo está bloqueado y no se lo vuelve a intentar hasta
entonces. Sin esto, cada consulta intentaría primero el modelo agotado, fallaría
y recién ahí pasaría al respaldo. Pasado ese tiempo se vuelve a probar con el
principal.

El estado vive en memoria y lo comparten todas las sesiones de la app (el cupo
es de la key, no de cada usuario). Si la app se reinicia se pierde, y el primer
intento vuelve a detectar el bloqueo; un rechazo por cupo no consume tokens.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

from langchain_groq import ChatGroq

from config import LLM_MODEL, LLM_MODELO_RESPALDO

# En orden de preferencia: se usa el primero que tenga cupo.
MODELOS = (LLM_MODEL, LLM_MODELO_RESPALDO)


def make_llm(modelo: str) -> ChatGroq:
    """Crea el cliente de un modelo. Es el único lugar donde se instancia."""
    return ChatGroq(model=modelo, temperature=0)


_llms = {modelo: make_llm(modelo) for modelo in MODELOS}
_bloqueado_hasta: dict[str, float] = {}
_candado = threading.Lock()


class CupoAgotado(Exception):
    """Todos los modelos agotaron su cupo diario."""

    def __init__(self, segundos: float):
        self.segundos = segundos
        super().__init__(
            f"Todos los modelos agotaron su cupo diario; "
            f"el primero se libera en {segundos:.0f} s"
        )


def es_limite_diario(texto: str) -> bool:
    """Indica si un error de la API es por el cupo diario (tokens o pedidos)."""
    return any(s in texto for s in ("tokens per day", "(TPD)", "requests per day", "(RPD)"))


def segundos_de_espera(mensaje: str) -> float | None:
    """Lee la espera que sugiere la API ("try again in 20m40.7s" -> 1240.7)."""
    m = re.search(
        r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?",
        mensaje,
    )
    if not m or not any(m.groups()):
        return None
    horas, minutos, segundos = m.groups()
    return int(horas or 0) * 3600 + int(minutos or 0) * 60 + float(segundos or 0)


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
            # El límite diario no se resuelve con reintentos: lo maneja
            # ``invocar`` pasando al modelo de respaldo.
            if es_limite_diario(mensaje):
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


def segundos_bloqueado(modelo: str) -> float:
    """Cuánto falta para volver a intentar ese modelo (0 si está disponible)."""
    return max(0.0, _bloqueado_hasta.get(modelo, 0.0) - time.time())


def invocar(construir: Callable[[ChatGroq], Any], entrada: Any) -> Any:
    """Invoca el primer modelo con cupo, en el orden de ``MODELOS``.

    ``construir`` arma, a partir del modelo, lo que se va a invocar. Para una
    respuesta de texto es el modelo tal cual (``lambda m: m``); para una salida
    estructurada, ``lambda m: m.with_structured_output(Schema, ...)``. Así el
    mismo pedido se puede repetir con el respaldo.

    Si todos los modelos están sin cupo, lanza ``CupoAgotado`` con la espera
    más corta.
    """
    for modelo in MODELOS:
        if segundos_bloqueado(modelo) > 0:
            continue  # disyuntor abierto: ni se intenta
        try:
            return invocar_con_reintentos(construir(_llms[modelo]).invoke, entrada)
        except Exception as e:  # noqa: BLE001
            if not es_limite_diario(str(e)):
                raise
            espera = segundos_de_espera(str(e)) or 60.0
            with _candado:
                _bloqueado_hasta[modelo] = time.time() + espera
            print(
                f"[modelos] {modelo} agotó su cupo diario; "
                f"no se lo usa por {max(1, round(espera / 60))} min"
            )
    raise CupoAgotado(min(segundos_bloqueado(m) for m in MODELOS))
