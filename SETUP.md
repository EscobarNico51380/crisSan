# SETUP — Cómo correr el proyecto en Windows

Guía paso a paso desde cero. Todos los comandos se ejecutan **parado en la
carpeta `tpIA/`** y desde **PowerShell** (la terminal por defecto de VSCode).

---

## Paso 1 — Conseguir la API key de Groq (gratis)

El proyecto usa **Groq** como proveedor del LLM. Es gratis y no pide tarjeta.

1. Entrá a **https://console.groq.com**
2. Creá la cuenta (podés entrar directo con Google o GitHub).
3. Una vez adentro, andá a **API Keys** en el menú lateral, o directo a
   **https://console.groq.com/keys**
4. Botón **"Create API Key"** → ponele un nombre (ej. `tp2-utn`) → **Submit**.
5. Te muestra la key **una sola vez**. Copiala ahora
   (arranca con `gsk_...`).

### Límites del plan gratuito

| Límite | Valor |
|---|---|
| Requests por minuto | 30 |
| Requests por día | ~1.000 |
| Tokens por día | 200.000 (por organización, no por modelo) |

> Alcanza de sobra para desarrollar y para la demo. Solo tené cuidado con
> `test_casos.py`, que dispara ~52 casos (por eso ya trae un `sleep(4)` entre
> cada uno para no pasarse de los 30 req/min).

---

## Paso 2 — Pegar la key en el `.env`

El archivo `.env` ya está creado en `tpIA/`. Abrilo en VSCode y pegá la key
después del `=`:

```
GROQ_API_KEY=gsk_tuKeyAcaSinComillasNiEspacios
```

O desde PowerShell, reemplazando por tu key:

```powershell
"GROQ_API_KEY=gsk_pega_tu_key_aca" | Out-File -FilePath .env -Encoding utf8
```

> ⚠️ **El `.env` está en `.gitignore`**, así que la key nunca se sube al repo.
> Nunca la pegues en un archivo `.py` ni la compartas por chat.

---

## Paso 3 — Entorno virtual e instalación

Un **entorno virtual** (venv) es una carpeta con su propio Python y sus propias
librerías, aislado del sistema. Así las versiones que pide este TP no chocan con
otros proyectos.

```powershell
# Crear el venv (una sola vez)
python -m venv .venv

# Instalar las dependencias
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tarda varios minutos: `sentence-transformers` arrastra **PyTorch**, que pesa
bastante.

### Activar el entorno (opcional pero cómodo)

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell te bloquea el script, habilitalo una vez para tu usuario:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Con el venv activado ya podés escribir `python main.py` en vez de la ruta larga.
Para salir: `deactivate`.

> **En VSCode:** `Ctrl+Shift+P` → "Python: Select Interpreter" → elegí el que
> diga `.venv`. Así el editor también usa ese entorno.

---

## Paso 4 — Orden en que conviene correr las cosas

Hay una dependencia importante: **la primera ejecución construye la base
vectorial** (descarga el modelo de embeddings ~1 GB, lee el PDF, genera los
vectores y los guarda en `data/chroma_db/`). Eso tarda unos minutos **una sola
vez**; después arranca en segundos.

### 4.1 — Verificar que la base relacional se crea (sin gastar API)

```powershell
.\.venv\Scripts\python.exe -c "import database; database.inicializar_base(); print(database.consultar_datos_poliza('AB123CD'))"
```

Esto **no llama al LLM**, así que no consume cupo. Si ves los datos de la póliza
de Carlos Perez, la capa SQLite anda.

### 4.2 — Construir e inspeccionar el RAG (sin gastar API)

```powershell
.\.venv\Scripts\python.exe -c "from rag import buscar_clausulas; print(buscar_clausulas('me cayo granizo sobre el auto'))"
```

Los embeddings corren **locales**, así que tampoco consume cupo de Groq. La
primera vez descarga el modelo y arma Chroma; sé paciente. Si ves cláusulas del
manual con su `[Sección | Pág]`, el RAG anda.

### 4.3 — El agente completo por consola (acá sí usa la API)

```powershell
.\.venv\Scripts\python.exe main.py
```

Probá en este orden para ver las distintas ramas del grafo:

| Escribí | Qué ejercita |
|---|---|
| `Hola` | rama `charla` |
| `¿Qué es la franquicia?` | rama `general` → RAG puro |
| `¿Qué plan tengo para la patente AB123CD?` | rama `consulta` → SQL + RAG |
| `¿Y cuándo vence?` | **memoria**: recuerda la patente del turno anterior |
| `Me cayó granizo sobre la AF987XX` | rama `siniestro` → dictamen + registro |
| `¿En qué estado está mi reclamo?` | consulta del reclamo recién creado |
| `Choqué contra un poste con la AF987XX` | dictamen **Rechazado** (C+ no cubre colisión parcial) |
| `salir` | terminar |

> El último caso es el más lindo para la demo: muestra que el sistema **rechaza
> con fundamento**, no que dice que sí a todo.

### 4.4 — La interfaz web (para el video)

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Abre el navegador solo. Es la que conviene grabar para el video.

### 4.5 — La batería de tests (opcional, consume bastante cupo)

```powershell
.\.venv\Scripts\python.exe test_casos.py
```

52 casos con `sleep(4)` entre cada uno → tarda ~5 minutos y consume cupo real.
Dejalo para cuando quieras evidencia de "funcionamiento del sistema" para la
defensa. Conviene guardar la salida:

```powershell
.\.venv\Scripts\python.exe test_casos.py | Tee-Object -FilePath resultados_tests.txt
```

---

## Resumen del orden

```
1. Key de Groq  →  2. pegarla en .env  →  3. venv + pip install
                          ↓
        4.1 database (sin API)  →  4.2 rag (sin API, tarda 1ª vez)
                          ↓
              4.3 main.py  →  4.4 streamlit  →  4.5 tests
```

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `No se encontró GROQ_API_KEY` | `.env` vacío o mal escrito | Revisá que no haya comillas ni espacios alrededor del `=` |
| Error 401 / `invalid_api_key` | Key mal copiada o revocada | Generá una nueva en console.groq.com/keys |
| Error 429 | Pasaste los 30 req/min | Esperá un minuto; el código ya reintenta con backoff |
| `tokens per day` | Agotaste los 200K diarios | Esperá al reset (medianoche UTC) |
| La 1ª consulta tarda muchísimo | Está bajando el modelo de embeddings | Normal, es una sola vez |
| `Activate.ps1 cannot be loaded` | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Querés rehacer el índice RAG | Cambiaste el PDF o el chunking | Borrá `data/chroma_db/` y volvé a correr |

---

## Qué se genera solo (y no se commitea)

```
tpIA/
├── .venv/              ← entorno virtual
├── .env                ← tu key (NUNCA se sube)
└── data/
    ├── seguros_tp.db   ← SQLite, se crea sola con datos sintéticos
    └── chroma_db/      ← base vectorial, se construye con el PDF
```

Todo eso está en `.gitignore`. Si clonás el repo en otra máquina, repetís estos
pasos y se regenera.


---

## Reproducir el entorno exacto de la entrega

`requirements.txt` no fija versiones, así que `pip install -r requirements.txt`
trae lo último que exista ese día. Eso ya rompió la app una vez (Streamlit 1.64
dejó de aceptar `page_icon=":car:"`). Para instalar las versiones con las que el
sistema fue efectivamente probado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Regenerar el lock después de cambiar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

---

## Reconstruir el índice RAG

Hay que hacerlo **cada vez que se toque el chunking** (`rag.py`), el PDF o el
modelo de embeddings. Si no, se sigue consultando el índice viejo y los cambios
no se ven:

```powershell
Remove-Item -Recurse -Force data\chroma_db
.\.venv\Scripts\python.exe -c "from rag import get_vector_store; get_vector_store(forzar=True)"
```

> Si `Remove-Item` falla con "archivo en uso", cerrá Streamlit y cualquier visor
> de SQLite/Chroma abierto: mantienen el archivo tomado.

Para inspeccionar cómo quedó un chunk puntual:

```powershell
.\.venv\Scripts\python.exe -c "from rag import _obtener_chunks; [print(c.metadata.get('page'), '|', c.page_content[:120]) for c in _obtener_chunks() if 'granizo' in c.page_content.lower()]"
```

---

## Limpiar los siniestros de prueba

Las corridas de prueba dejan filas en `siniestros`, y esas filas **se sirven como
precedentes** a los dictámenes siguientes. Si quedaron dictámenes equivocados,
conviene borrarlos antes de grabar la demo:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3;c=sqlite3.connect('data/seguros_tp.db');c.execute('DELETE FROM siniestros');c.commit();print('siniestros borrados')"
```

---

## Modelos de Groq disponibles

| Modelo | Uso | Acepta imágenes |
|---|---|---|
| `openai/gpt-oss-120b` | el que usa el agente | ❌ no |
| `openai/gpt-oss-20b` | más rápido, mismo cupo | ❌ no |
| `qwen/qwen3.8-27b` | **visión** | ✅ **sí** |
| `whisper-large-v3` | transcripción de audio | — |

Para la funcionalidad de interpretar imágenes (pendiente del alcance original)
hay que usar `qwen/qwen3.8-27b`: `gpt-oss` no es multimodal. Costo medido: entre
**1.300 y 1.800 tokens por imagen** según el tamaño, sobre un cupo de 200.000
tokens diarios.
