"""San Cris: instala todo la primera vez y abre el chat.

Uso:
    python iniciar.py                  instala (si hace falta) y abre el chat web
    python iniciar.py --consola        igual, pero abre el chat en la terminal
    python iniciar.py --solo-instalar  instala y termina, sin abrir el chat
    python iniciar.py --si             no pide confirmación antes de instalar

Usa solo la biblioteca estándar, porque corre con el Python del sistema antes de
que exista el entorno virtual. Todo lo que instala queda dentro de esta carpeta:
para desinstalar, alcanza con borrarla.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
DATA = RAIZ / "data"
MODELOS = DATA / "modelos"  # el modelo de embeddings, en vez de la caché del usuario
MARCA = DATA / ".instalado"
LOCK = RAIZ / "requirements-lock.txt"
ENV = RAIZ / ".env"
PUERTO = 8501
# Repositorio oficial de PyTorch, versión solo-CPU (se usa en Linux).
PYTORCH_CPU = "https://download.pytorch.org/whl/cpu"

PYTHON_PROBADO = (3, 13)
PYTHON_COMPATIBLE = ((3, 11), (3, 13))  # rango en el que existen todas las dependencias

if os.name == "nt":
    PYTHON_VENV = VENV / "Scripts" / "python.exe"
else:
    PYTHON_VENV = VENV / "bin" / "python"

# La consola de Windows no siempre usa UTF-8: sin esto, los acentos fallan.
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def titulo(texto: str) -> None:
    print(f"\n=== {texto} ===")


def entorno() -> dict:
    """Variables para los procesos hijos: el modelo de embeddings va a data/."""
    env = dict(os.environ)
    env["HF_HOME"] = str(MODELOS)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def correr(comando: list, **kwargs) -> None:
    """Ejecuta un comando mostrando su salida; si falla, corta con un mensaje."""
    resultado = subprocess.run(comando, cwd=RAIZ, env=entorno(), **kwargs)
    if resultado.returncode != 0:
        print(
            "\nAlgo falló en el paso anterior (ver el mensaje de arriba).\n"
            "Podés volver a ejecutar `python iniciar.py`: retoma desde donde quedó."
        )
        sys.exit(resultado.returncode)


def en_venv(*args: str, **kwargs) -> None:
    """Corre este mismo script con el Python del entorno virtual."""
    correr([str(PYTHON_VENV), str(Path(__file__).resolve()), *args], **kwargs)


# ---------------------------------------------------------------------------
# Instalación (corre con el Python del sistema)
# ---------------------------------------------------------------------------
def requisito_del_lock(paquete: str) -> str:
    """Devuelve "paquete==versión" tal como figura en el lock."""
    for linea in LOCK.read_text(encoding="utf-8").splitlines():
        if linea.startswith(f"{paquete}=="):
            return linea.split(";")[0].strip()
    raise SystemExit(f"{paquete} no figura en {LOCK.name}")


def instalado() -> bool:
    return MARCA.exists() and PYTHON_VENV.exists()


def explicar_y_confirmar(sin_preguntar: bool) -> None:
    version = ".".join(map(str, sys.version_info[:3]))
    print(
        f"""
San Cris todavía no está instalado. Esto es lo que se va a hacer,
TODO dentro de esta carpeta:

    {RAIZ}

  1. Crear un entorno de Python aislado (carpeta .venv) e instalar las
     librerías del proyecto: LangChain, LangGraph, PyTorch, Streamlit, etc.
     Ocupa unos 1,5 GB.
  2. Descargar el modelo de embeddings multilingual-e5-base, que se usa
     para buscar en el manual de la póliza (carpeta data/modelos).
     Ocupa unos 1,1 GB.
  3. Crear la base de datos de prueba y el índice del manual (carpeta
     data). Ocupa unos 3 MB.

  - No instala nada fuera de esta carpeta: usa el Python que ya tenés
    (versión {version}) y no guarda copias en la caché del sistema. La
    única excepción es Streamlit, que crea una carpeta .streamlit en tu
    usuario con dos archivos de menos de 100 bytes.
  - No necesita permisos de administrador ni cambia la configuración de
    tu computadora.
  - Descarga unos 1,5 GB y ocupa unos 2,6 GB. Según la conexión, tarda
    entre 5 y 20 minutos. Al terminar, se abre el chat en el navegador.
  - Durante la instalación pueden aparecer avisos (una versión nueva de
    pip, descargas de Hugging Face sin token, "symlinks" en Windows): no
    son errores y se pueden ignorar.
  - Para desinstalar: borrá esta carpeta.
"""
    )
    if sin_preguntar:
        return
    respuesta = input("¿Continuar? [S/n] ").strip().lower()
    if respuesta not in ("", "s", "si", "sí", "y", "yes"):
        print("Instalación cancelada. No se hizo ningún cambio.")
        sys.exit(0)


def verificar_python() -> None:
    actual = sys.version_info[:2]
    desde, hasta = PYTHON_COMPATIBLE
    if desde <= actual <= hasta:
        return
    print(
        f"\nAtención: estás usando Python {actual[0]}.{actual[1]}. El sistema se "
        f"probó con Python {PYTHON_PROBADO[0]}.{PYTHON_PROBADO[1]}, y algunas "
        f"librerías (como PyTorch) pueden no estar disponibles para tu versión.\n"
        f"Se recomienda instalar Python {PYTHON_PROBADO[0]}.{PYTHON_PROBADO[1]} "
        f"desde https://www.python.org/downloads/"
    )
    if input("¿Intentar igual? [s/N] ").strip().lower() not in ("s", "si", "sí", "y"):
        sys.exit(1)


def instalar() -> None:
    titulo("1/3 · Entorno de Python")
    if PYTHON_VENV.exists():
        print("Ya existe (.venv); se reutiliza.")
    else:
        print("Creando .venv ...")
        venv.create(VENV, with_pip=True)

    titulo("2/3 · Librerías (la parte más larga)")
    # No se actualiza pip: el que trae Python 3.11-3.13 alcanza, y así no hay
    # ninguna descarga sin versión fijada.
    pip = [str(PYTHON_VENV), "-m", "pip", "install", "--no-cache-dir"]
    if sys.platform.startswith("linux"):
        # En Linux, el PyTorch de PyPI es la versión para GPU, con varios GB de
        # librerías de NVIDIA que el sistema no usa. Se instala la versión CPU
        # desde el repositorio oficial de PyTorch, en un paso aparte:
        #   --index-url  usa SOLO ese repositorio (no lo mezcla con PyPI);
        #   --no-deps    trae de ahí SOLO estos dos paquetes; sus dependencias
        #                salen de PyPI en el paso siguiente.
        # Así cada paquete viene de un único origen. Mezclar repositorios en un
        # mismo comando (--extra-index-url) permite que un paquete llegue del
        # repositorio equivocado: así fue el ataque a PyTorch de diciembre de 2022.
        print("PyTorch, versión CPU, desde el repositorio oficial de PyTorch...")
        correr(pip + [
            "--index-url", PYTORCH_CPU, "--no-deps",
            requisito_del_lock("torch"), requisito_del_lock("torchvision"),
        ])
    print("Librerías del proyecto, desde PyPI...")
    correr(pip + ["-r", str(LOCK)])
    # Verifica que todo lo instalado sea compatible entre sí.
    correr([str(PYTHON_VENV), "-m", "pip", "check"])

    titulo("3/3 · Modelo de embeddings, base de datos e índice del manual")
    en_venv("--preparar")

    DATA.mkdir(exist_ok=True)
    MARCA.write_text(
        f"Instalado el {datetime.now():%Y-%m-%d %H:%M} con Python "
        f"{'.'.join(map(str, sys.version_info[:3]))}\n",
        encoding="utf-8",
    )
    print("\nInstalación completa.")


# ---------------------------------------------------------------------------
# Pasos que corren con el Python del entorno virtual
# ---------------------------------------------------------------------------
def preparar() -> None:
    """Crea la base, arma el índice y descarga el modelo de embeddings."""
    sys.path.insert(0, str(RAIZ))
    import database
    from rag import buscar_clausulas, get_vector_store

    database.inicializar_base()
    print("Base de datos de prueba: lista.")
    print("Descargando el modelo de embeddings y armando el índice del manual...")
    get_vector_store()
    # Una búsqueda de prueba garantiza que el modelo quedó descargado, aunque
    # el índice ya viniera armado.
    buscar_clausulas("granizo")
    print("Índice del manual: listo.")


def verificar_key() -> None:
    """Se asegura de que haya una key de Groq válida en .env; si no, la pide."""
    sys.path.insert(0, str(RAIZ))
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ENV)
    key = os.getenv("GROQ_API_KEY", "").strip()
    for intento in range(3):
        if not key:
            print(
                "\nFalta la API key de Groq (el modelo de lenguaje). Se consigue gratis,\n"
                "sin tarjeta, en https://console.groq.com/keys (empieza con gsk_)."
            )
            key = input("Pegá la key y apretá Enter: ").strip()
        try:
            # Listar los modelos valida la key sin consumir tokens.
            Groq(api_key=key).models.list()
        except Exception as e:  # noqa: BLE001
            print(f"La key no funcionó ({type(e).__name__}). Probá de nuevo.")
            key = ""
            continue
        guardar_key(key)
        print("API key de Groq: válida.")
        return
    print("No se pudo validar la key. Revisá el archivo .env y volvé a intentar.")
    sys.exit(1)


def guardar_key(key: str) -> None:
    """Escribe la key en .env, conservando el resto del archivo."""
    lineas = []
    if ENV.exists():
        lineas = [
            l for l in ENV.read_text(encoding="utf-8").splitlines()
            if not l.strip().startswith("GROQ_API_KEY")
        ]
    lineas.append(f"GROQ_API_KEY={key}")
    ENV.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Abrir el chat
# ---------------------------------------------------------------------------
def abrir_chat_web() -> None:
    url = f"http://localhost:{PUERTO}"
    titulo("Abriendo el chat")
    # Modo "headless": Streamlit no frena la terminal pidiendo un email la
    # primera vez. El navegador lo abre este script cuando el servidor responde.
    proceso = subprocess.Popen(
        [
            str(PYTHON_VENV), "-m", "streamlit", "run", "app.py",
            "--server.headless", "true", "--server.port", str(PUERTO),
            # Solo accesible desde esta computadora: no expone el chat a la red
            # y el firewall de Windows no pregunta nada.
            "--server.address", "localhost",
        ],
        cwd=RAIZ,
        env=entorno(),
    )
    for _ in range(90):
        if proceso.poll() is not None:
            print(f"\nEl chat no pudo arrancar. ¿Hay otra ventana del chat abierta en el puerto {PUERTO}?")
            sys.exit(1)
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except OSError:
            time.sleep(1)
    webbrowser.open(url)
    print(
        f"\nEl chat está abierto en {url}\n"
        "Si el navegador no se abrió solo, copiá esa dirección.\n"
        "La primera consulta tarda unos segundos más (carga el modelo de embeddings).\n"
        "Para cerrar el chat: Ctrl+C en esta terminal."
    )
    try:
        proceso.wait()
    except KeyboardInterrupt:
        proceso.terminate()
        print("\nChat cerrado.")


def abrir_chat_consola() -> None:
    titulo("Chat por consola")
    try:
        subprocess.run([str(PYTHON_VENV), "main.py"], cwd=RAIZ, env=entorno())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
def main() -> None:
    args = set(sys.argv[1:])

    # Pasos internos, que este mismo script lanza con el Python del entorno.
    if "--preparar" in args:
        preparar()
        return
    if "--verificar-key" in args:
        verificar_key()
        return

    print("San Cris — Asistente de seguros (TP2 de Inteligencia Artificial, UTN FRRO)")
    if not instalado():
        explicar_y_confirmar(sin_preguntar="--si" in args)
        verificar_python()
        instalar()
    en_venv("--verificar-key")

    if "--solo-instalar" in args:
        print("\nListo. Para abrir el chat: python iniciar.py")
    elif "--consola" in args:
        abrir_chat_consola()
    else:
        abrir_chat_web()


if __name__ == "__main__":
    main()
