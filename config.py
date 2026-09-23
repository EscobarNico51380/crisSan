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
# Free tier de Groq: 200.000 tokens/día POR ORGANIZACIÓN (no por modelo).
# - "openai/gpt-oss-120b" (120B): mejor razonamiento, 500 tps.
# - "openai/gpt-oss-20b" (20B): más rápido (1000 tps), mismo cupo diario.
LLM_MODEL = "openai/gpt-oss-120b"
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


def ensure_dirs() -> None:
    """Crea las carpetas de datos si todavía no existen."""
    DATA_DIR.mkdir(exist_ok=True)
