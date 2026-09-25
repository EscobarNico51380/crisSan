# Guía interna del proyecto

Para el equipo y para cualquier asistente de IA que trabaje en este repositorio.
El README es para quien **usa** el sistema; esta guía es para quien lo
**modifica**. La arquitectura, las decisiones y las dificultades resueltas están
en `DOCUMENTACION.md`; lo que falta hacer, en `PENDIENTES.md`.

---

## Qué es

San Cris: asistente de seguros de auto (TP2 de Inteligencia Artificial, UTN
FRRO, comisión 502). Grafo de estado de LangGraph que clasifica el mensaje,
consulta la póliza (SQLite), busca en el manual (RAG híbrido sobre Chroma),
emite un dictamen de cobertura con el LLM, lo registra y responde.

## Mapa de archivos

| Archivo | Qué hace |
|---|---|
| `agent.py` | El grafo: nodos, ruteo, prompts (`PROMPT_*`) y controles deterministas (vigencia, pago, patente, DNI) |
| `llm.py` | Único lugar donde se crean los modelos. Modelo principal, respaldo y disyuntor. **Toda llamada al LLM pasa por `llm.invocar`**. También el modelo de visión (`describir_foto`, `preparar_foto`) |
| `rag.py` | Carga del PDF, chunking estructural, embeddings e5, Chroma y búsqueda híbrida (BM25 + semántica) |
| `database.py` | Esquema SQLite, datos de prueba y consultas (siempre parametrizadas) |
| `tools.py` | Las herramientas `@tool` que usa el grafo |
| `costos.py` | Tokens y costo estimado de cada turno |
| `config.py` | Modelos, precios, rutas y parámetros (chunking, historial) |
| `app.py` / `main.py` | Chat web (Streamlit) y chat por consola |
| `iniciar.py` | Instalación y arranque con un solo comando (solo biblioteca estándar) |
| `test_casos.py` | 52 casos que pasan por el agente completo (consume cupo) |
| `ejemplos/fotos/` | Fotos de ejemplo (granizo, parabrisas, choque) con sus créditos |
| `notebooks_clases/` | Notebooks de la cátedra pasados a `.py`, como referencia |

---

## Correrlo mientras se desarrolla

La forma más simple es la misma que usa el profesor:

```
python iniciar.py              # instala si hace falta y abre el chat web
python iniciar.py --consola    # chat en la terminal
```

A mano, con el entorno ya instalado (Windows, desde la carpeta del proyecto):

```powershell
$env:HF_HOME = "data\modelos"          # el modelo de embeddings está ahí (ver abajo)
.\.venv\Scripts\streamlit.exe run app.py
.\.venv\Scripts\python.exe main.py
```

> **Ojo con `HF_HOME`.** `iniciar.py` guarda el modelo de embeddings en
> `data/modelos`. Si se corre la app a mano sin esa variable, se descarga otra
> copia (1,1 GB) en la caché del usuario.

Después de cambiar código, reiniciar Streamlit: el agente queda en caché
(`st.cache_resource`).

## La API key y el cupo

- Va en `.env` (`GROQ_API_KEY=gsk_...`), que git ignora. **Nunca commitear keys.**
  Se consigue gratis en https://console.groq.com/keys.
- El plan gratuito da **200.000 tokens por día, por modelo y por cuenta**, contados
  sobre las últimas 24 horas. Dos keys de la misma cuenta comparten el cupo.
- Un dictamen usa ~9.000 tokens; una consulta, ~4.000. Una batería de pruebas
  puede gastar la mitad del cupo diario.
- Cuando el principal (`gpt-oss-120b`) se agota, `llm.py` pasa solo al respaldo
  (`gpt-oss-20b`) y no reintenta el principal hasta que se libere. Un rechazo
  por cupo no consume tokens.

| Modelo | Uso | Imágenes |
|---|---|---|
| `openai/gpt-oss-120b` | Principal | No |
| `openai/gpt-oss-20b` | Respaldo | No |
| `qwen/qwen3.8-27b` | Candidato para interpretar imágenes (pendiente) | Sí (mínimo 32×32 px; ~1.300-1.800 tokens por foto) |

---

## Probar

**Sin gastar API:**

```powershell
# Base de datos
.\.venv\Scripts\python.exe -c "import database; database.inicializar_base(); print(database.consultar_datos_poliza('AB123CD'))"
# RAG (los embeddings son locales)
.\.venv\Scripts\python.exe -c "from rag import buscar_clausulas; print(buscar_clausulas('granizo'))"
```

La lógica que no depende del LLM (vigencia, disyuntor, mensajes de error,
costos) se prueba llamando a sus funciones con datos armados o con modelos
simulados. Ver ejemplos en `DOCUMENTACION.md`.

**Con API:** las baterías que se usaron (6 preguntas del GNC, 7 de los autos de
prueba, seguimiento, memoria) están descritas en `DOCUMENTACION.md`, con las
preguntas literales y los resultados. Se corren en el momento con un script
aparte; no se guardan scripts de prueba en el repo.

Antes de una batería, **vaciar la tabla `siniestros`**: los reclamos quedan como
precedentes y sesgan los dictámenes siguientes.

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3;c=sqlite3.connect('data/seguros_tp.db');c.execute('DELETE FROM siniestros');c.commit()"
```

---

## Trampas conocidas

- **Si se toca el chunking (`rag.py`), hay que rearmar el índice** (borrar
  `data/chroma_db`) y repetir la batería del GNC: el chunking fue la causa del
  bug más largo del proyecto.
- **El lock y `torchvision`.** `pip freeze` en Windows no incluye `torchvision`
  (solo hace falta en Linux). Si se regenera `requirements-lock.txt`, volver a
  agregar al final la línea `torchvision==0.29.0; sys_platform == "linux"`.
- **Todo lo que se pueda decidir con código no se le pide al LLM:** la patente y
  el DNI se extraen con regex; la vigencia y el pago se comparan en
  `motivo_de_rechazo_administrativo`. El LLM interpreta el lenguaje y el manual.
- **No crear clientes de `ChatGroq` fuera de `llm.py`**: se perdería el respaldo.
- **Las exclusiones del manual dependen de su encabezado** (ej. el GNC excluye
  robo e incendio, no todo). Por eso cada fragmento lleva el título de su lista,
  y las exclusiones por características del vehículo se buscan aparte.
- **Errores `No module named 'torchvision'` en la consola de Streamlit:** son
  ruido, no rompen nada. Los genera el vigilante de archivos de Streamlit al
  inspeccionar módulos de imágenes de `transformers` que no se usan. Ver
  `PENDIENTES.md`.
- **Las instrucciones del modelo de visión (`PROMPT_FOTO`) son muy sensibles:**
  una oración más hizo que volviera a confundir granizo con nieve. Si se
  cambian, repetir la prueba con las tres fotos de ejemplo (ver `DOCUMENTACION.md`).
- **El modelo de visión limita la salida a 1.000 tokens por minuto** en el plan
  gratuito: no subir `VISION_MAX_TOKENS` ni prender su razonamiento.
- **Una tarea por llamada al LLM:** la comparación foto-relato falló como
  regla y como campo del dictamen, y funcionó como llamada aparte.
- **Consola de Windows:** usar `PYTHONIOENCODING=utf-8` si fallan los acentos.

---

## Cómo trabaja el equipo

1. **Primero los problemas, después las features.**
2. Antes de cambiar algo: explicar qué pasa, por qué y cómo se arreglaría, y
   esperar el OK del equipo.
3. **Un commit por tema**, con un mensaje que explique el porqué. Los push los
   hace una persona del equipo.
4. **Cada cambio se documenta en `DOCUMENTACION.md`**: una fila en la tabla de
   dificultades (problema, causa, solución) y, si es grande, una sección con la
   prueba y sus resultados, citando las preguntas literales.
5. **`PENDIENTES.md` es la única lista de tareas**: al terminar algo, marcarlo ✅
   con su commit.
6. Código, comentarios y documentación en **español rioplatense**.

## La entrega

Se entrega un **zip**, no el repositorio. `README.md` es la puerta de entrada
para el profesor. Al final se agrega un `.env` con una key de una cuenta de Groq
nueva, dedicada a la entrega (sin tarjeta), que se revoca después de la nota.
