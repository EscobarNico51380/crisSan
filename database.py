"""Capa de acceso a datos: base relacional SQLite con datos sintéticos.

Modela el negocio asegurador (clientes, vehículos, pólizas y siniestros)
y expone las funciones de consulta/escritura que después consumen las tools
del agente. Acá NO hay nada de IA: es el "mock de base de datos" que pide
el enunciado, y la fuente principal para tomar decisiones.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from config import SQLITE_PATH, ensure_dirs


def get_conn() -> sqlite3.Connection:
    """Devuelve una conexión con filas accesibles por nombre de columna."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema y datos sintéticos
# ---------------------------------------------------------------------------
def inicializar_base() -> None:
    """Crea el schema y carga datos sintéticos (idempotente)."""
    ensure_dirs()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clientes (
            dni TEXT PRIMARY KEY,
            nombre_completo TEXT NOT NULL,
            email TEXT NOT NULL,
            telefono TEXT,
            domicilio_localidad TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehiculos (
            patente TEXT PRIMARY KEY,
            marca TEXT NOT NULL,
            modelo TEXT NOT NULL,
            anio INTEGER NOT NULL,
            categoria TEXT NOT NULL,
            uso TEXT NOT NULL,
            tiene_gnc INTEGER DEFAULT 0,
            rastreador_satelital INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS polizas (
            numero_poliza TEXT PRIMARY KEY,
            dni_titular TEXT NOT NULL,
            patente TEXT NOT NULL,
            producto TEXT NOT NULL,
            suma_asegurada REAL NOT NULL,
            tipo_franquicia TEXT,
            valor_franquicia REAL DEFAULT 0.0,
            estado_pago TEXT NOT NULL,
            fecha_inicio TEXT NOT NULL,
            fecha_vencimiento TEXT NOT NULL,
            FOREIGN KEY (dni_titular) REFERENCES clientes (dni),
            FOREIGN KEY (patente) REFERENCES vehiculos (patente)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS siniestros (
            id_siniestro INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_poliza TEXT NOT NULL,
            fecha_ocurrencia TEXT NOT NULL,
            fecha_denuncia TEXT NOT NULL,
            tipo_siniestro TEXT NOT NULL,
            hubo_lesionados INTEGER NOT NULL,
            descripcion_usuario TEXT NOT NULL,
            estado_reclamo TEXT DEFAULT 'Iniciado',
            dictamen_ia TEXT,
            FOREIGN KEY (numero_poliza) REFERENCES polizas (numero_poliza)
        )
        """
    )

    # --- Datos sintéticos ---
    cur.execute(
        "INSERT OR IGNORE INTO clientes VALUES "
        "('39123456', 'Carlos Perez', 'carlos@mail.com', '3415551234', 'Rosario')"
    )

    # Auto 1: Focus 2024, particular, con rastreador.
    cur.execute(
        "INSERT OR IGNORE INTO vehiculos VALUES "
        "('AB123CD', 'Ford', 'Focus', 2024, 'Automovil', 'Particular', 0, 1)"
    )
    # Póliza 1: Todo Riesgo con franquicia fija.
    cur.execute(
        "INSERT OR IGNORE INTO polizas VALUES "
        "('POL-1001', '39123456', 'AB123CD', 'D c/fran', 25000000.0, 'Fija', 460000.0, "
        "'Al dia', '2026-01-01', '2027-01-01')"
    )

    # Auto 2: Peugeot 208 2012 con GNC, sin rastreador.
    cur.execute(
        "INSERT OR IGNORE INTO vehiculos VALUES "
        "('AF987XX', 'Peugeot', '208', 2012, 'Automovil', 'Particular', 1, 0)"
    )
    # Póliza 2: Terceros Completos Avanzado (C+), sin franquicia.
    cur.execute(
        "INSERT OR IGNORE INTO polizas VALUES "
        "('POL-1002', '39123456', 'AF987XX', 'C+', 12000000.0, 'Sin Franquicia', 0.0, "
        "'Al dia', '2026-06-01', '2027-06-01')"
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Consultas y escritura
# ---------------------------------------------------------------------------
def consultar_datos_poliza(patente: str) -> str:
    """Devuelve un resumen legible de la póliza asociada a una patente."""
    patente = patente.strip().upper()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.numero_poliza, c.nombre_completo, c.domicilio_localidad,
               v.marca, v.modelo, v.uso, v.tiene_gnc, v.rastreador_satelital,
               p.producto, p.suma_asegurada, p.tipo_franquicia,
               p.valor_franquicia, p.estado_pago, p.fecha_vencimiento
        FROM polizas p
        JOIN clientes c ON p.dni_titular = c.dni
        JOIN vehiculos v ON p.patente = v.patente
        WHERE p.patente = ?
        """,
        (patente,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return f"No se encontró ninguna póliza asociada a la patente '{patente}'."

    return (
        f"Patente: {patente}\n"
        f"Póliza: {row['numero_poliza']} | Titular: {row['nombre_completo']} "
        f"({row['domicilio_localidad']})\n"
        f"Vehículo: {row['marca']} {row['modelo']} (Uso: {row['uso']}, "
        f"GNC: {'sí' if row['tiene_gnc'] else 'no'}, "
        f"Rastreador: {'sí' if row['rastreador_satelital'] else 'no'})\n"
        f"Producto: {row['producto']} | Suma asegurada: ${row['suma_asegurada']:,.2f}\n"
        f"Franquicia: {row['tipo_franquicia']} (${row['valor_franquicia']:,.2f})\n"
        f"Estado de pago: {row['estado_pago']} | Vencimiento: {row['fecha_vencimiento']}"
    )


# El estado inicial del reclamo depende del dictamen del siniestro: un
# siniestro rechazado no debe quedar como "Iniciado".
ESTADO_POR_DICTAMEN = {
    "Cubierto": "Iniciado",
    "Rechazado": "Rechazado",
    "Requiere peritaje": "En análisis",
    "Asistencia 24": "Derivado a Asistencia 24",
}


def registrar_siniestro(
    patente: str,
    tipo_siniestro: str,
    fecha_ocurrencia: str,
    hubo_lesionados: int,
    descripcion: str,
    dictamen: str,
) -> str:
    """Inserta una denuncia de siniestro y devuelve el número de reclamo."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT numero_poliza FROM polizas WHERE patente = ?",
        (patente.strip().upper(),),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return f"No se encontró póliza para la patente '{patente}'."

    numero_poliza = row["numero_poliza"]
    fecha_denuncia = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    estado = ESTADO_POR_DICTAMEN.get(dictamen, "Iniciado")

    cur.execute(
        """
        INSERT INTO siniestros
            (numero_poliza, fecha_ocurrencia, fecha_denuncia, tipo_siniestro,
             hubo_lesionados, descripcion_usuario, estado_reclamo, dictamen_ia)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            numero_poliza,
            fecha_ocurrencia,
            fecha_denuncia,
            tipo_siniestro,
            hubo_lesionados,
            descripcion,
            estado,
            dictamen,
        ),
    )
    conn.commit()
    id_siniestro = cur.lastrowid
    conn.close()

    alerta = ""
    if hubo_lesionados == 1:
        alerta = (
            "\nALERTA CRÍTICA: Hay lesionados. Recordá contactar a Asistencia 24 "
            "al 0810-222-8887 de inmediato."
        )

    return (
        f"Siniestro registrado con número de reclamo #{id_siniestro}. "
        f"Estado: {estado}.{alerta}"
    )


def consultar_siniestros_previos(limite: int = 5) -> str:
    """Devuelve los siniestros previos registrados con su dictamen.

    Se usan como precedentes para fundamentar un nuevo dictamen (la fuente
    de "resoluciones previas similares" del alcance del TP).
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.tipo_siniestro, s.dictamen_ia, s.estado_reclamo, p.producto
        FROM siniestros s
        JOIN polizas p ON s.numero_poliza = p.numero_poliza
        WHERE s.dictamen_ia IS NOT NULL
        ORDER BY s.id_siniestro DESC
        LIMIT ?
        """,
        (limite,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "No hay siniestros previos registrados para usar como precedentes."

    partes = []
    for r in rows:
        partes.append(
            f"- {r['tipo_siniestro']} | Producto {r['producto']} | "
            f"Dictamen: {r['dictamen_ia']} | Estado: {r['estado_reclamo']}"
        )
    return "\n".join(partes)


def consultar_polizas_por_dni(dni: str) -> str:
    """Devuelve todas las pólizas asociadas a un DNI de titular."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.numero_poliza, p.patente, p.producto, p.estado_pago,
               c.nombre_completo, v.marca, v.modelo
        FROM polizas p
        JOIN clientes c ON p.dni_titular = c.dni
        JOIN vehiculos v ON p.patente = v.patente
        WHERE c.dni = ?
        """,
        (dni.strip(),),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return f"No se encontraron pólizas para el DNI '{dni}'."

    partes = []
    for r in rows:
        partes.append(
            f"- Patente {r['patente']} | Póliza {r['numero_poliza']} | "
            f"{r['marca']} {r['modelo']} | Producto {r['producto']} | "
            f"Pago: {r['estado_pago']}"
        )
    return f"Titular: {rows[0]['nombre_completo']}\n" + "\n".join(partes)


def consultar_estado_reclamo(numero_reclamo: str) -> str:
    """Devuelve el estado y dictamen de un reclamo por su número."""
    try:
        numero = int(numero_reclamo)
    except (TypeError, ValueError):
        return f"No se encontró ningún reclamo con el número '{numero_reclamo}'."

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id_siniestro, s.numero_poliza, s.tipo_siniestro,
               s.dictamen_ia, s.estado_reclamo, s.fecha_denuncia, p.patente
        FROM siniestros s
        JOIN polizas p ON s.numero_poliza = p.numero_poliza
        WHERE s.id_siniestro = ?
        """,
        (numero,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return f"No se encontró ningún reclamo con el número #{numero_reclamo}."

    return (
        f"Reclamo #{row['id_siniestro']} | Póliza {row['numero_poliza']} "
        f"(patente {row['patente']})\n"
        f"Tipo: {row['tipo_siniestro']} | Dictamen: {row['dictamen_ia']}\n"
        f"Estado: {row['estado_reclamo']} | Denunciado: {row['fecha_denuncia']}"
    )
