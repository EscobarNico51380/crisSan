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
            fundamento_ia TEXT,
            FOREIGN KEY (numero_poliza) REFERENCES polizas (numero_poliza)
        )
        """
    )
    # Migración: el motivo del dictamen se guarda desde el 23/09/2026. Las
    # bases creadas antes no tienen la columna; se agrega sin tocar los datos.
    columnas = {c["name"] for c in cur.execute("PRAGMA table_info(siniestros)")}
    if "fundamento_ia" not in columnas:
        cur.execute("ALTER TABLE siniestros ADD COLUMN fundamento_ia TEXT")

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

    # --- Autos de prueba adicionales ------------------------------------------
    # Cada uno aísla UNA variable. Con los dos autos de arriba no alcanzaba: el
    # único con GNC era también el único C+, y no se podía saber cuál de las
    # dos cosas causaba un rechazo.
    cur.executemany(
        "INSERT OR IGNORE INTO clientes VALUES (?, ?, ?, ?, ?)",
        [
            ("30555777", "Laura Gomez", "laura@mail.com", "3425550000", "Santa Fe"),
            ("28444666", "Martin Diaz", "martin@mail.com", "3415559876", "Rosario"),
        ],
    )
    cur.executemany(
        "INSERT OR IGNORE INTO vehiculos VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # patente, marca, modelo, año, categoría, uso, GNC, rastreador
            ("AC456BD", "Volkswagen", "Gol Trend", 2018, "Automovil", "Particular", 0, 0),
            ("AD789EF", "Chevrolet", "Cruze", 2022, "Automovil", "Particular", 1, 1),
            ("AE321GH", "Fiat", "Uno", 2008, "Automovil", "Particular", 0, 0),
            ("AG654JK", "Renault", "Kangoo", 2015, "Utilitario", "Particular", 0, 0),
            ("AH111LM", "Toyota", "Etios", 2019, "Automovil", "Particular", 0, 0),
            ("AJ222NP", "Honda", "Fit", 2014, "Automovil", "Particular", 0, 0),
        ],
    )
    cur.executemany(
        "INSERT OR IGNORE INTO polizas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # C+ SIN GNC: el robo se cubre. Contrasta con AF987XX (C+ con GNC).
            ("POL-1003", "30555777", "AC456BD", "C+", 9000000.0,
             "Sin Franquicia", 0.0, "Al dia", "2026-03-01", "2027-03-01"),
            # Todo riesgo CON GNC: la exclusión se aplica aun con el mejor plan.
            ("POL-1004", "30555777", "AD789EF", "D c/fran", 30000000.0,
             "Fija", 460000.0, "Al dia", "2026-02-01", "2027-02-01"),
            # Plan A (solo responsabilidad civil): no cubre daños al propio auto.
            ("POL-1005", "30555777", "AE321GH", "A", 4000000.0,
             "Sin Franquicia", 0.0, "Al dia", "2026-05-01", "2027-05-01"),
            # Plan B (daños totales): cubre robo total, no daños parciales.
            ("POL-1006", "28444666", "AG654JK", "B", 11000000.0,
             "Sin Franquicia", 0.0, "Al dia", "2026-04-01", "2027-04-01"),
            # Póliza IMPAGA: cualquier siniestro se rechaza.
            ("POL-1007", "28444666", "AH111LM", "C+", 14000000.0,
             "Sin Franquicia", 0.0, "Impaga", "2026-01-15", "2027-01-15"),
            # Póliza VENCIDA (junio de 2026): cualquier siniestro se rechaza.
            ("POL-1008", "28444666", "AJ222NP", "C+", 8000000.0,
             "Sin Franquicia", 0.0, "Al dia", "2025-06-01", "2026-06-01"),
        ],
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
    fundamento: str = "",
) -> str:
    """Inserta una denuncia de siniestro y devuelve el número de reclamo.

    ``fundamento`` es el motivo del dictamen: se guarda para poder explicarlo
    cuando el cliente consulta el reclamo.
    """
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
             hubo_lesionados, descripcion_usuario, estado_reclamo, dictamen_ia,
             fundamento_ia)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            fundamento,
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


# Aviso que encabeza los precedentes cuando hubo que ampliar la busqueda.
AVISO_PRECEDENTES_AMPLIADOS = (
    "(Sin precedentes del mismo tipo/producto; se listan los mas recientes.)"
)

# Consulta base de precedentes. Las condiciones extra se agregan con AND.
_SQL_PRECEDENTES = """
    SELECT s.tipo_siniestro, s.dictamen_ia, s.estado_reclamo, p.producto
    FROM siniestros s
    JOIN polizas p ON s.numero_poliza = p.numero_poliza
    WHERE s.dictamen_ia IS NOT NULL
"""
_ORDEN_PRECEDENTES = " ORDER BY s.id_siniestro DESC LIMIT ?"


def consultar_siniestros_previos(
    limite: int = 5,
    tipo_siniestro: str = "",
    patente: str = "",
) -> str:
    """Devuelve siniestros previos con su dictamen, para usarlos como precedentes.

    Filtros OPCIONALES (ambos vacios = comportamiento historico, los ultimos N):

    - ``tipo_siniestro``: coincidencia parcial por tipo (granizo, robo, choque).
    - ``patente``: restringe a polizas del MISMO producto que el de esa patente.
      Es lo que hace comparable a un precedente: no sirve citar la resolucion de
      un Todo Riesgo para dictaminar un Terceros Completos.

    Si el filtro no devuelve nada se cae al listado sin filtrar (encabezado con
    ``AVISO_PRECEDENTES_AMPLIADOS``), para que el dictamen nunca se quede sin
    referencias.
    """
    conn = get_conn()
    cur = conn.cursor()

    condiciones: list[str] = []
    valores: list[str] = []

    if tipo_siniestro.strip():
        condiciones.append("LOWER(s.tipo_siniestro) LIKE ?")
        valores.append(f"%{tipo_siniestro.strip().lower()}%")

    if patente.strip():
        fila = cur.execute(
            "SELECT producto FROM polizas WHERE patente = ?",
            (patente.strip().upper(),),
        ).fetchone()
        if fila:
            condiciones.append("p.producto = ?")
            valores.append(fila["producto"])

    rows = []
    hubo_filtro = bool(condiciones)
    if hubo_filtro:
        sql = _SQL_PRECEDENTES + " AND " + " AND ".join(condiciones) + _ORDEN_PRECEDENTES
        rows = cur.execute(sql, (*valores, limite)).fetchall()

    # Fallback: si no hubo filtro, o si el filtro no encontro nada.
    ampliado = False
    if not rows:
        ampliado = hubo_filtro
        rows = cur.execute(
            _SQL_PRECEDENTES + _ORDEN_PRECEDENTES, (limite,)
        ).fetchall()

    conn.close()

    if not rows:
        return "No hay siniestros previos registrados para usar como precedentes."

    partes = []
    if ampliado:
        partes.append(AVISO_PRECEDENTES_AMPLIADOS)
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


def _fecha_legible(texto: str | None) -> str:
    """'2026-09-23' -> '23/09/2026'; '2026-09-23 22:02:42' -> '23/09/2026 a las 22:02'."""
    if not texto:
        return "sin dato"
    try:
        if len(texto) > 10:
            return datetime.strptime(texto[:16], "%Y-%m-%d %H:%M").strftime("%d/%m/%Y a las %H:%M")
        return datetime.strptime(texto, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return texto


def obtener_reclamo(numero_reclamo: str) -> dict | None:
    """Devuelve todos los datos de un reclamo, o ``None`` si no existe."""
    try:
        numero = int(numero_reclamo)
    except (TypeError, ValueError):
        return None

    conn = get_conn()
    row = conn.execute(
        """
        SELECT s.id_siniestro, s.numero_poliza, s.tipo_siniestro,
               s.fecha_ocurrencia, s.fecha_denuncia, s.hubo_lesionados,
               s.descripcion_usuario, s.dictamen_ia, s.fundamento_ia,
               s.estado_reclamo, p.patente, p.producto,
               v.marca, v.modelo, c.nombre_completo
        FROM siniestros s
        JOIN polizas p ON s.numero_poliza = p.numero_poliza
        JOIN vehiculos v ON p.patente = v.patente
        JOIN clientes c ON p.dni_titular = c.dni
        WHERE s.id_siniestro = ?
        """,
        (numero,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def formatear_reclamo(datos: dict, markdown: bool = False) -> str:
    """Arma la ficha de un reclamo, en texto plano o en Markdown para el chat."""
    lesionados = "con lesionados" if datos["hubo_lesionados"] else "sin lesionados"
    motivo = datos["fundamento_ia"] or (
        "no quedó registrado (el reclamo es anterior a que se guardara el motivo)"
    )
    campos = [
        (
            "Vehículo",
            f"{datos['marca']} {datos['modelo']}, patente {datos['patente']} "
            f"(póliza {datos['numero_poliza']}, producto {datos['producto']})",
        ),
        ("Titular", datos["nombre_completo"]),
        (
            "Siniestro",
            f"{datos['tipo_siniestro']}, ocurrido el "
            f"{_fecha_legible(datos['fecha_ocurrencia'])}, {lesionados}",
        ),
        ("Relato del cliente", f'"{datos["descripcion_usuario"]}"'),
        ("Denunciado", f"el {_fecha_legible(datos['fecha_denuncia'])}"),
        ("Dictamen", datos["dictamen_ia"]),
        ("Motivo", motivo),
        ("Estado", datos["estado_reclamo"]),
    ]
    if markdown:
        lineas = [f"**Reclamo #{datos['id_siniestro']}**", ""]
        lineas += [f"- **{clave}:** {valor}" for clave, valor in campos]
    else:
        lineas = [f"Reclamo #{datos['id_siniestro']}"]
        lineas += [f"{clave}: {valor}" for clave, valor in campos]
    return "\n".join(lineas)


def consultar_estado_reclamo(numero_reclamo: str) -> str:
    """Devuelve la ficha completa de un reclamo por su número.

    Incluye el vehículo, qué pasó y cuándo, el relato del cliente, el dictamen
    con su motivo y el estado.
    """
    datos = obtener_reclamo(numero_reclamo)
    if not datos:
        return f"No se encontró ningún reclamo con el número #{numero_reclamo}."
    return formatear_reclamo(datos)


def vehiculo_tiene_gnc(patente: str) -> bool:
    """Indica si el vehículo de esa patente está adaptado para GNC.

    El manual excluye a estos vehículos de algunas coberturas, pero el cliente
    no suele mencionarlo al denunciar ("me robaron el auto"). El agente usa este
    dato para buscar esas exclusiones por su cuenta.
    """
    conn = get_conn()
    fila = conn.execute(
        "SELECT tiene_gnc FROM vehiculos WHERE patente = ?",
        (patente.strip().upper(),),
    ).fetchone()
    conn.close()
    return bool(fila and fila["tiene_gnc"])


def obtener_vigencia(patente: str) -> dict | None:
    """Devuelve las fechas de vigencia y el estado de pago de la póliza de una patente.

    Lo usa el agente para controlar, sin el LLM, que la póliza estuviera
    vigente y paga en la fecha del siniestro.
    """
    conn = get_conn()
    fila = conn.execute(
        """
        SELECT numero_poliza, fecha_inicio, fecha_vencimiento, estado_pago
        FROM polizas
        WHERE patente = ?
        """,
        (patente.strip().upper(),),
    ).fetchone()
    conn.close()
    return dict(fila) if fila else None
