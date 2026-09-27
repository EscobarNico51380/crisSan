"""Herramientas del agente (decoradas con @tool).

Cada tool encapsula una operación concreta sobre la base relacional o la
base vectorial. El grafo de estado las invoca en el orden que corresponde
según la intención del usuario.
"""

from __future__ import annotations

from langchain_core.tools import tool

import database
from rag import SEPARADOR_FRAGMENTOS, buscar_clausulas, buscar_exclusiones

# Cuántos fragmentos del manual recibe el subagente ReAct por búsqueda.
FRAGMENTOS_REACT = 6

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


def exclusiones_del_vehiculo(patente: str) -> str:
    """Las exclusiones del manual por GNC si ese vehículo lo tiene; si no, ""."""
    if not patente or not database.vehiculo_tiene_gnc(patente):
        return ""
    return buscar_exclusiones(CONSULTA_EXCLUSIONES_GNC, CLAVES_GNC)


@tool
def consultar_datos_poliza(patente: str) -> str:
    """Consulta en la base relacional la póliza de un vehículo por patente.

    Devuelve titular, producto contratado, suma asegurada, franquicia,
    estado de pago y vencimiento.
    """
    return database.consultar_datos_poliza(patente)


@tool
def consultar_clausulas_poliza(consulta: str, patente: str = "") -> str:
    """Busca en el manual (Chroma) las cláusulas relevantes a una consulta.

    Parámetros:
    - consulta: la duda o descripción del evento en lenguaje natural.
    - patente: si la pregunta es sobre la cobertura de un vehículo en
      particular, su patente. Así se suman las exclusiones que dependen del
      vehículo (por ejemplo, las del GNC), que la búsqueda sola no trae.
    """
    # Los FRAGMENTOS_REACT mejores (vienen ordenados por relevancia). La
    # búsqueda completa trae ~10 (~2.500 tokens) y el ReAct puede buscar más
    # de una vez: con dos o tres búsquedas un pedido pasaba el límite del plan
    # gratuito (8.000 tokens por minuto) y Groq lo rechazaba con un 413.
    fragmentos = buscar_clausulas(consulta).split(SEPARADOR_FRAGMENTOS)
    resultado = SEPARADOR_FRAGMENTOS.join(fragmentos[:FRAGMENTOS_REACT])
    exclusiones = exclusiones_del_vehiculo(patente)
    if exclusiones:
        resultado += (
            "\n\n--- EXCLUSIONES POR CARACTERÍSTICAS DEL VEHÍCULO "
            f"(la patente {patente.strip().upper()} tiene GNC) ---\n{exclusiones}"
        )
    return resultado


@tool
def registrar_siniestro(
    patente: str,
    tipo_siniestro: str,
    fecha_ocurrencia: str,
    hubo_lesionados: int,
    descripcion: str,
    dictamen: str,
    fundamento: str = "",
    descripcion_foto: str = "",
) -> str:
    """Registra una denuncia de siniestro en la base relacional.

    Devuelve el número de reclamo generado.
    """
    return database.registrar_siniestro(
        patente=patente,
        tipo_siniestro=tipo_siniestro,
        fecha_ocurrencia=fecha_ocurrencia,
        hubo_lesionados=hubo_lesionados,
        descripcion=descripcion,
        dictamen=dictamen,
        fundamento=fundamento,
        descripcion_foto=descripcion_foto,
    )


@tool
def consultar_siniestros_previos(
    limite: int = 5,
    tipo_siniestro: str = "",
    patente: str = "",
) -> str:
    """Consulta los siniestros previos registrados con sus dictámenes.

    Se usan como precedentes para fundamentar un nuevo dictamen.

    Parámetros opcionales para acotarlos a casos comparables:
    - tipo_siniestro: filtra por tipo (granizo, robo, choque...).
    - patente: restringe a pólizas del mismo producto que el de esa patente.
    """
    return database.consultar_siniestros_previos(limite, tipo_siniestro, patente)


@tool
def consultar_polizas_por_dni(dni: str) -> str:
    """Devuelve las pólizas de un titular a partir de su DNI."""
    return database.consultar_polizas_por_dni(dni)


@tool
def consultar_estado_reclamo(numero_reclamo: str) -> str:
    """Consulta el estado y dictamen de un reclamo por su número."""
    return database.consultar_estado_reclamo(numero_reclamo)


# ---------------------------------------------------------------------------
# Herramientas expuestas al subagente ReAct
# ---------------------------------------------------------------------------
@tool
def listar_reclamos(patente: str = "", dni: str = "") -> str:
    """Lista los reclamos (siniestros denunciados) de un vehículo o de un titular.

    Pasá la patente para los reclamos de ese vehículo, o el DNI para los de
    todos los vehículos del titular. Devuelve número, fecha, tipo, dictamen y
    estado de cada uno. Para el detalle de uno, usá consultar_estado_reclamo.
    """
    return database.listar_reclamos(patente, dni)


@tool
def calcular_pago_cliente(patente: str, costo_reparacion: float) -> str:
    """Calcula cuánto paga el cliente y cuánto la aseguradora por una reparación.

    Aplica la franquicia y la suma asegurada de la póliza de esa patente.
    Usala SIEMPRE que haya que hacer esta cuenta, en vez de calcularla vos.
    Supone que el daño está cubierto: no decide la cobertura.

    Parámetros:
    - patente: la patente del vehículo.
    - costo_reparacion: el costo total de la reparación, en pesos.
    """
    return database.calcular_pago_cliente(patente, costo_reparacion)


# SOLO LECTURA. Las herramientas con efectos secundarios (registrar_siniestro)
# NO se le entregan al LLM: quedan reservadas al camino determinístico, para no
# arriesgar registros duplicados ni decisiones de cobertura no auditables.
HERRAMIENTAS_LECTURA = [
    consultar_datos_poliza,
    consultar_clausulas_poliza,
    consultar_polizas_por_dni,
    consultar_estado_reclamo,
    listar_reclamos,
    calcular_pago_cliente,
]
