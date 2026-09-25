"""Herramientas del agente (decoradas con @tool).

Cada tool encapsula una operación concreta sobre la base relacional o la
base vectorial. El grafo de estado las invoca en el orden que corresponde
según la intención del usuario.
"""

from __future__ import annotations

from langchain_core.tools import tool

import database
from rag import buscar_clausulas


@tool
def consultar_datos_poliza(patente: str) -> str:
    """Consulta en la base relacional la póliza de un vehículo por patente.

    Devuelve titular, producto contratado, suma asegurada, franquicia,
    estado de pago y vencimiento.
    """
    return database.consultar_datos_poliza(patente)


@tool
def consultar_clausulas_poliza(consulta: str) -> str:
    """Busca en el manual (Chroma) las cláusulas relevantes a una consulta.

    Parámetros:
    - consulta: la duda o descripción del evento en lenguaje natural.
    """
    return buscar_clausulas(consulta)


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
