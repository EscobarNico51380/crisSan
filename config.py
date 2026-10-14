"""Configuración central del proyecto.

Carga las variables de entorno desde el ``.env`` y centraliza paths y
constantes. Todo es relativo al root del proyecto para que funcione igual
en local y en Google Colab (subiendo la carpeta completa).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Root del proyecto (la carpeta donde vive este archivo).
ROOT = Path(__file__).resolve().parent

# Cargamos el .env desde el root, sin depender del working directory.
load_dotenv(ROOT / ".env")

# --- Credenciales -----------------------------------------------------------
# langchain-groq lee GROQ_API_KEY automáticamente del entorno,
# pero la exponemos acá por claridad y para validar que exista.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Modelos ----------------------------------------------------------------
# Free tier de Groq: 200.000 tokens/día POR MODELO y por organización (cuenta).
# Verificado: con el cupo de gpt-oss-120b agotado, gpt-oss-20b respondía con la
# misma key. Dos keys de la misma cuenta comparten el cupo; otra cuenta trae el
# suyo. El límite cuenta los tokens de las últimas 24 horas.
# - "openai/gpt-oss-120b" (120B): mejor razonamiento, 500 tps.
# - "openai/gpt-oss-20b" (20B): más rápido (1000 tps), con su propio cupo.
LLM_MODEL = "openai/gpt-oss-120b"
# Respaldo: se usa solo cuando el principal agota su cupo diario (ver llm.py).
# Probado con las dos baterías: coincide con el principal en todos los casos
# claros (DOCUMENTACION.md, prueba con el modelo chico).
LLM_MODELO_RESPALDO = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# --- Almacenamiento ---------------------------------------------------------
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_PATH = DATA_DIR / "seguros_tp.db"
PDF_PATH = ROOT / "poliza_sanCristobal.pdf"

# --- Chunking (RAG) ---------------------------------------------------------
# Chunks más grandes + más solapamiento para que títulos como "10) Granizo"
# no queden separados de su contenido (degradaba la recuperación semántica).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300


# --- Historial que ve el clasificador -----------------------------------------
# Mensajes previos que recibe para entender las preguntas de seguimiento
# ("¿y cuándo vence?"). Cada uno se recorta a HISTORIAL_CLASIFICADOR_CHARS para
# no gastar tokens en respuestas largas con cláusulas.
HISTORIAL_CLASIFICADOR = 4
HISTORIAL_CLASIFICADOR_CHARS = 300


# --- Memoria de la conversación: ventana + resumen ---------------------------
# Al llegar a HISTORIAL_UMBRAL_RESUMEN mensajes, los más viejos se comprimen en
# un resumen y se borran del estado; quedan los últimos
# HISTORIAL_MENSAJES_A_CONSERVAR completos. Cada turno suma dos mensajes
# (pregunta y respuesta), así que con 12 y 6 se resume una vez cada 3 turnos.
HISTORIAL_UMBRAL_RESUMEN = 12
HISTORIAL_MENSAJES_A_CONSERVAR = 6
# Tope de caracteres por mensaje al armar el pedido de resumen.
HISTORIAL_RESUMEN_CHARS = 800


# --- Costo estimado de las consultas -----------------------------------------
# Precios del plan pago de Groq, en USD por millón de tokens. Fuente:
# https://console.groq.com/docs/model/openai/gpt-oss-120b (consultado el
# 2026-09-23). Con el plan gratuito el costo real es cero: se usan para estimar
# cuánto costaría. Si se suma un modelo (por ejemplo, uno de visión), agregar
# su precio acá; si no, sus tokens se cuentan pero no suman costo.
PRECIOS_USD_POR_MILLON = {
    "openai/gpt-oss-120b": {"entrada": 0.15, "entrada_cache": 0.075, "salida": 0.60},
    # https://console.groq.com/docs/model/openai/gpt-oss-20b (2026-09-23)
    "openai/gpt-oss-20b": {"entrada": 0.075, "entrada_cache": 0.037, "salida": 0.30},
    # https://console.groq.com/docs/model/qwen/qwen3.8-27b (2026-09-25)
    "qwen/qwen3.8-27b": {"entrada": 0.80, "entrada_cache": 0.80, "salida": 4.00},
}


# --- Plazo de denuncia -----------------------------------------------------
# El manual pide denunciar cualquier siniestro dentro de las 72 horas de
# ocurrido el hecho (pág. 25 y 26); la Ley de Seguros 17.418 (art. 46) fija 3
# días, bajo pena de perder la indemnización. Como la fecha del siniestro se
# conoce sin hora, se cuenta en días.
PLAZO_DENUNCIA_DIAS = 3


# --- Fotos ------------------------------------------------------------------
# Modelo de visión: el único de Groq que acepta imágenes. Solo DESCRIBE la foto;
# el dictamen lo sigue emitiendo el modelo principal.
LLM_MODELO_VISION = "qwen/qwen3.8-27b"
# En el plan gratuito, este modelo limita la salida a 1.000 tokens por minuto y
# rechaza de entrada los pedidos que podrían pasarse: hay que fijar un máximo.
# Una descripción usa ~110 tokens.
VISION_MAX_TOKENS = 400
# La foto se achica antes de mandarla: sube más rápido y gasta menos.
FOTO_LADO_MAXIMO = 1024
# Tamaño máximo del adjunto en el chat, en MB.
FOTO_TAMANO_MAXIMO_MB = 10


def ensure_dirs() -> None:
    """Crea las carpetas de datos si todavía no existen."""
    DATA_DIR.mkdir(exist_ok=True)
