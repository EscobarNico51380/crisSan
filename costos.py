"""Costo estimado de las consultas, a partir de los tokens que usa el LLM.

Cada llamada al LLM informa cuántos tokens de entrada y de salida usó. Durante
un turno, ``UsageMetadataCallbackHandler`` (de LangChain) los va sumando por
modelo; acá se multiplican por los precios de ``config.py``.

Con el plan gratuito de Groq el costo real es cero: lo que se calcula es cuánto
costaría la misma consulta en el plan pago. Los embeddings corren localmente y
no tienen costo.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import PRECIOS_USD_POR_MILLON

ACLARACION = (
    "Estimado con los precios del plan pago de Groq. "
    "Con el plan gratuito, el costo real es cero."
)


@dataclass(frozen=True)
class Consumo:
    """Tokens usados y su costo estimado, para un turno o una conversación."""

    tokens_entrada: int = 0
    tokens_salida: int = 0
    usd: float = 0.0
    modelos_sin_precio: tuple[str, ...] = ()

    def __add__(self, otro: Consumo) -> Consumo:
        return Consumo(
            self.tokens_entrada + otro.tokens_entrada,
            self.tokens_salida + otro.tokens_salida,
            self.usd + otro.usd,
            tuple(sorted(set(self.modelos_sin_precio) | set(otro.modelos_sin_precio))),
        )


def consumo_de(uso_por_modelo: dict) -> Consumo:
    """Calcula el consumo a partir de lo que juntó ``UsageMetadataCallbackHandler``.

    ``uso_por_modelo`` mapea el nombre del modelo a sus tokens. Si un modelo no
    tiene precio cargado en ``config.py``, sus tokens se cuentan igual pero no
    suman costo, y se informa en ``modelos_sin_precio``.
    """
    total = Consumo()
    for modelo, uso in uso_por_modelo.items():
        entrada = uso.get("input_tokens", 0) or 0
        salida = uso.get("output_tokens", 0) or 0
        # Groq cobra a mitad de precio la parte de la entrada que ya tenía en
        # caché. Si la API no lo informa, se asume cero.
        en_cache = (uso.get("input_token_details") or {}).get("cache_read", 0) or 0

        precio = PRECIOS_USD_POR_MILLON.get(modelo)
        if precio is None:
            total = total + Consumo(entrada, salida, 0.0, (modelo,))
            continue
        usd = (
            (entrada - en_cache) * precio["entrada"]
            + en_cache * precio["entrada_cache"]
            + salida * precio["salida"]
        ) / 1_000_000
        total = total + Consumo(entrada, salida, usd)
    return total


def _miles(n: int) -> str:
    """12345 -> "12.345" (separador de miles argentino)."""
    return f"{n:,}".replace(",", ".")


def formatear(c: Consumo) -> str:
    """Ej: "4.172 tokens (3.799 de entrada + 373 de salida) · US$ 0,0008"."""
    usd = f"{c.usd:.4f}".replace(".", ",")
    texto = (
        f"{_miles(c.tokens_entrada + c.tokens_salida)} tokens "
        f"({_miles(c.tokens_entrada)} de entrada + {_miles(c.tokens_salida)} de salida)"
        f" · US$ {usd}"
    )
    if c.modelos_sin_precio:
        texto += f" (sin precio cargado para: {', '.join(c.modelos_sin_precio)})"
    return texto
