# Documentación Técnica — TP2: Agente Conversacional de Seguros

> **Sistema experto** de seguros automotor basado en **RAG + agentes inteligentes**,
> construido con LangChain, LangGraph, Groq, ChromaDB y HuggingFace.

---

## 1. Descripción del sistema

"San Cris" es un asistente virtual para una aseguradora (San Cristóbal) capaz de:

- Responder preguntas sobre el contenido de una póliza (**RAG** sobre el manual).
- **Determinar si un siniestro está cubierto o no** y fundamentarlo.
- Consultar el estado, plan y datos de una póliza (por patente o por DNI).
- Registrar siniestros en una base relacional y emitir número de reclamo.
- Usar **resoluciones previas** (siniestros históricos) como precedentes.
- Mantener **memoria conversacional** entre turnos.

El sistema es un **sistema experto funcional**: recupera información relevante,
toma decisiones y dialoga en lenguaje natural, cumpliendo los objetivos del enunciado.

---

## 2. Caso de negocio

Los seguros están llenos de "letra chica" (condiciones, exclusiones, franquicias).
Los clientes no leen los PDFs y no entienden si un evento está cubierto.

El agente resuelve esto:
1. El cliente describe su situación en lenguaje natural.
2. El sistema recupera las cláusulas relevantes del manual (RAG).
3. Cruza la información con la póliza real del cliente (base relacional).
4. Emite un **dictamen** (cubierto / rechazado / requiere peritaje) con fundamento.

La fuente de verdad es el manual de San Cristóbal (`poliza_sanCristobal.pdf`,
27 páginas) y una base SQLite con datos sintéticos.

---

## 3. Arquitectura general

```
Usuario (Streamlit / consola)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              GRAFO DE ESTADO (LangGraph)            │
│                                                     │
│  clasificar ──┬── responder (charla)                │
│   (LLM)       ├── consultar_clausulas (general)     │
│   + regex     ├── listar_polizas (DNI)              │
│  patente/DNI  ├── pedir_patente                     │
│               └── consultar_poliza → cláusulas →    │
│                       emitir_dictamen → registrar   │
│                                                     │
│  Memoria: MemorySaver + thread_id                   │
└──────────┬───────────────────────┬──────────────────┘
           │                       │
     ┌─────▼─────┐          ┌──────▼──────┐
     │  SQLite    │          │  ChromaDB   │
     │ (pólizas,  │          │ (manual     │
     │ siniestros)│          │  vectorizado)│
     └───────────┘          └─────────────┘
```

**Flujo principal del dictamen (siniestro):**

```
extraer patente → consultar póliza (SQL) → buscar cláusulas (RAG híbrido)
        → consultar precedentes → emitir dictamen (LLM) → registrar siniestro
```

---

## 4. Stack tecnológico

| Componente | Tecnología | Por qué |
|---|---|---|
| Orquestación | **LangGraph** (`StateGraph`) | Control explícito del flujo (grafo de estado), no un ReAct a ciegas |
| Framework | **LangChain** (`@tool`, retriever, loaders) | Requisito del enunciado |
| LLM | **Groq** — `openai/gpt-oss-120b` | Free tier: 200K tokens/día (por organización), 30 req/min. Rápido (500 tps) |
| Structured output | `json_schema` (nativo de Groq) | Fuerza el schema del lado del servidor |
| Base vectorial | **ChromaDB** (espacio coseno) | Persistencia local, simple |
| Embeddings | **`intfloat/multilingual-e5-base`** | Entrenado específicamente para retrieval |
| Retrieval | **Híbrido** (BM25 + semántico) | `EnsembleRetriever` + `rank_bm25` |
| Base relacional | **SQLite** | Mock de la fuente de datos transaccional |
| Frontend | **Streamlit** | Interfaz de chat rápida |
| Config | `python-dotenv` + `.env` | Secretos fuera del código |
| PDF | `pypdf` + `PyPDFLoader` | Carga del manual |

---

## 5. Componentes (archivos)

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Carga `.env`, paths, modelos y constantes centralizadas |
| `database.py` | Schema SQLite (clientes, vehículos, pólizas, siniestros) + consultas |
| `rag.py` | Pipeline RAG: carga PDF, chunking, embeddings, Chroma, retrieval híbrido |
| `tools.py` | Herramientas del agente decoradas con `@tool` |
| `agent.py` | Grafo de estado LangGraph (nodos, routing, prompts, memoria) |
| `main.py` | Loop interactivo por consola |
| `app.py` | Interfaz web Streamlit (chat + memoria por sesión) |

---

## 6. Herramientas del agente (`@tool`)

| Tool | Input | Acción |
|---|---|---|
| `consultar_datos_poliza` | patente | Consulta SQL: póliza, titular, producto, franquicia, estado |
| `consultar_clausulas_poliza` | consulta | Búsqueda RAG híbrida en el manual |
| `consultar_polizas_por_dni` | dni | Lista las pólizas de un titular |
| `consultar_estado_reclamo` | número de reclamo | Devuelve el estado y dictamen de un reclamo |
| `consultar_siniestros_previos` | (límite) | Devuelve siniestros históricos con su dictamen (precedentes) |
| `registrar_siniestro` | patente, tipo, fecha, lesionados, dictamen | INSERT en `siniestros` + número de reclamo |

---

## 7. El grafo de estado (LangGraph)

**Estado** (`EstadoAgente`): `messages`, `intencion`, `patente`, `dni`,
`reclamo`, `datos_poliza`, `clausulas`, `dictamen`, `siniestro_registrado`,
`reclamo_resultado`.

**Nodos:**

| Nodo | Qué hace |
|---|---|
| `clasificar` | Extrae patente/DNI con **regex** + clasifica intención con LLM (`json_schema`) |
| `pedir_patente` | Pide la patente si falta |
| `listar_polizas` | Lista las pólizas de un DNI y pide elegir |
| `consultar_reclamo` | Consulta el estado de un reclamo por su número |
| `consultar_poliza` | Llama a la tool de SQL |
| `consultar_clausulas` | RAG híbrido sobre el manual |
| `emitir_dictamen` | LLM emite determinación + precedentes |
| `registrar` | INSERT del siniestro en SQLite |
| `responder` | Genera la respuesta final en lenguaje natural |

**Routing** (señales explícitas del mensaje actual > intención del LLM):
```
reclamo (si se menciona) → consultar_reclamo
dni/patente (en el mensaje) → listar_polizas / consultar_poliza
general                  → consultar_clausulas
charla                   → responder
consulta/siniestro con patente recordada → consultar_poliza
(ninguno)                → pedir_patente
```

**Memoria:** `MemorySaver` (checkpointer) + `thread_id` por sesión. Los campos
`patente`, `dni` y `reclamo` se **preservan entre turnos**; los campos "por
turno" (dictamen, cláusulas, etc.) se resetean en `clasificar`.

---

## 8. Pipeline RAG

1. **Carga**: `PyPDFLoader` sobre el manual (27 páginas).
2. **Limpieza**: se excluye el índice (páginas 0-2) porque concentra keywords
   y contamina la búsqueda semántica.
3. **Chunking estructural**: se divide por los **ítems numerados** del manual
   (`1)`, `2)`, `10) Granizo`...) con `re.split(r"(?=^\d+\))")`. Ítems muy
   largos se subdividen con `RecursiveCharacterTextSplitter` (1000 / overlap 300).
   *Motivo*: el chunking por caracteres separaba el título del contenido.
4. **Etiquetado por sección** según la página (A/B/C/D/General).
5. **Embeddings**: `multilingual-e5-base` con prefijos `query:` / `passage:`.
6. **Retrieval híbrido**: `BM25Retriever` (léxico) + `as_retriever` (semántico)
   fusionados con `EnsembleRetriever` (pesos 0.5/0.5).

---

## 9. Decisiones de diseño (con tradeoffs)

1. **Groq en vez de Gemini** → el free tier de Gemini limita a 20 req/día;
   Groq da 1.000 req/día. Costo: modelo distinto (GPT-OSS).
2. **`json_schema` en vez de `function_calling`** → más robusto ante inputs
   ambiguos. Costo: menos flexible que function calling.
3. **e5 en vez de MiniLM/mpnet** → los modelos de "similitud" no discriminan
   términos raros de dominio. Costo: e5 requiere prefijos query/passage.
4. **Retrieval híbrido (BM25 + semántico)** → combina exactitud léxica y
   comprensión semántica. Costo: complejidad extra y pesos a tunear.
5. **Grafo de estado explícito en vez de ReAct** → control total del flujo y
   del dictamen. Costo: más código que un `create_react_agent`.
6. **Extracción determinística de patente/DNI (regex)** → robusta, no depende
   del LLM. Costo: menos flexible para formatos no previstos.

---

## 10. Dificultades encontradas y soluciones

> **Esta sección responde directamente al criterio de evaluación
> "Identificación de dificultades".**

| # | Problema | Causa raíz | Solución |
|---|---|---|---|
| 1 | Gemini agotaba quota (429) | Free tier = 20 req/día | Migrar a **Groq** (1.000 req/día) |
| 2 | `gemini-2.5-flash` deprecado (404) | Google retiró modelos 2.5 | Usar `gemini-3.6-flash`, luego Groq |
| 3 | Gemini 3.x devuelve contenido como lista de bloques | Nuevo formato con "thought signature" | Helper `extraer_texto()` |
| 4 | `function_calling` alucinaba nombres de campo (`{"consulta":...}` en vez de `{"intencion":...}`) | LLM produce tool calls malformados en inputs ambiguos | `method="json_schema"` + fallback heurístico |
| 5 | El LLM no mapeaba "patente AB123CD" → "póliza POL-1001" | El output de la tool no incluía la patente | Hacer el output de la tool **auto-descriptivo** |
| 6 | Dictamen decía "cubierto" para choque con póliza C+ | El LLM no conocía la taxonomía de coberturas | Agregar mapeo producto→cobertura al prompt |
| 7 | La memoria no funcionaba entre turnos | `nodo_clasificar` sobreescribía `patente` con "" | Preservar patente del estado anterior |
| 8 | El número de reclamo nunca aparecía | `siniestro_registrado` no estaba en el schema del estado | Agregarlo al `EstadoAgente` |
| 9 | "granizo" no se recuperaba | MiniLM/mpnet no discriminan términos raros | **e5** + prefijos query/passage |
| 10 | El header "10) Granizo" quedaba separado de su cuerpo | Chunking por caracteres cortaba mal | **Chunking estructural** por ítems |
| 11 | BM25 no matcheaba "granizo" (score 0) | Tokenizer naive: sensible a mayúsculas y puntuación | `preprocess_func` propio (lowercase + regex) |
| 12 | El índice/TOC contaminaba la búsqueda | Concentra todas las keywords | Excluir páginas del índice |
| 13 | El LLM inventaba emails/teléfonos | Alucinación cuando el contexto no alcanza | Endurecer el prompt anti-alucinación |
| 14 | Alucinaba portales/apps/emails al preguntar por el estado del reclamo | No existía tool para consultar el reclamo | Tool `consultar_estado_reclamo` + anti-alucinación en responder |
| 15 | El patente recordado "secuestraba" el routing (consultar reclamo iba a consultar_poliza) | El patente persistido se chequeaba antes que las señales del mensaje | Priorizar señales explícitas del mensaje actual |
| 16 | Respuesta vacía del modelo | Flakiness del LLM / rate limits | Fallback en `nodo_responder` |
| 17 | Clasificador impreciso ("datos de la patente" → general) | Prompt de clasificación vago | Prompt con definiciones + ejemplos |
| 18 | Siniestros rechazados quedaban "Iniciado" | `estado_reclamo` hardcodeado | Estado según dictamen (`ESTADO_POR_DICTAMEN`) |
| 19 | `dni` no se preservaba entre turnos | Faltaba preservarlo como patente/reclamo | Preservar dni del estado anterior |
| 20 | Streamlit no abría: `OSError [WinError 123]` en bucle | Streamlit 1.64 dejó de interpretar el shortcode `page_icon=":car:"` y lo buscaba como archivo estático; `:` es inválido en nombres de archivo en Windows | Emoji literal `"🚗"` |
| 21 | El entorno no era reproducible | `requirements.txt` sin versiones fijadas: `pip install` trae lo último de cada día (así se coló el bug 20) | `requirements-lock.txt` con las 139 versiones probadas |
| 22 | Exclusión por GNC aplicada fuera de su alcance | El chunking partía por ítems numerados y cada ítem quedaba **sin el encabezado que le da ámbito**. El chunk recuperado decía sólo "5) Cuando el vehículo haya sido adaptado para GNC", sin indicar que pertenece a *Exclusiones para Robo o Hurto* | Propagar el encabezado de lista a cada ítem (numerado y con viñetas) al indexar |

---

## 11. Cómo ejecutar

```bash
# 1. Crear entorno e instalar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configurar la API key de Groq en .env
#    GROQ_API_KEY=tu_key  (obtenela en https://console.groq.com/keys)

# 3. Interfaz web
.venv/bin/streamlit run app.py

#    O consola
.venv/bin/python main.py
```

La base SQLite y Chroma se generan solas en `data/` la primera vez.

---

## 12. Funcionalidades implementadas

- ✅ Chat conversacional con memoria entre turnos.
- ✅ RAG sobre el manual (retrieval híbrido BM25 + embeddings).
- ✅ Dictamen de cobertura con fundamento (cubierto/rechazado/peritaje).
- ✅ Registro de siniestros + número de reclamo.
- ✅ Consulta por patente y por DNI.
- ✅ Consulta del estado de un reclamo por su número.
- ✅ Resoluciones previas como precedentes.
- ✅ Interfaz web (Streamlit) y consola.
- ✅ Manejo de rate limits con reintentos.

## 13. Limitaciones y mejoras futuras

- La tabla de límites (pág 5) a veces rankea alto (falso positivo menor).
- El modelado de GNC no distingue "declarado vs no declarado".
- **Interpretación de imágenes (multimodal)** — pendiente del alcance original.
  Verificado: `openai/gpt-oss-*` **no** acepta imágenes; `qwen/qwen3.8-27b` sí,
  y está disponible en el free tier. Costo medido: 1.300–1.800 tokens por
  imagen sobre un cupo de 200.000 diarios (~40-50 dictámenes con foto por día).
- **La consulta al RAG no incorpora los atributos del vehículo** (ver §14).

---

## 14. Estado del bug del GNC y próximo paso

> Sección de traspaso: acá está lo que quedó a medio camino, con el diagnóstico
> hecho para que quien siga no tenga que repetirlo.

### Qué se arregló

El chunking ahora propaga a cada ítem el encabezado de la lista que lo gobierna,
tanto para listas numeradas (`N)`) como con viñetas (`•`). El manual tiene
**tres** listas distintas que excluyen vehículos con GNC:

| Página | Alcance de la exclusión |
|---|---|
| 10 | General — *"excepto que la cobertura incluya alguno de los puntos enumerados"* |
| 15 | Sólo **Incendio** |
| 16 | Sólo **Robo o Hurto** |

Antes, los tres chunks llegaban al LLM sin ese encabezado y la exclusión se leía
como absoluta.

**Resultado medido** (batería de 6 casos cruzando tipo de siniestro × GNC):

| Caso | Antes del fix | Después |
|---|---|---|
| Granizo / vehículo con GNC | Rechazado (falso negativo) | **Requiere peritaje** |
| Robo / vehículo con GNC | Cubierto (falso positivo) | Cubierto — *sin cambios* |
| Los otros 4 casos | correctos | correctos |

La exclusión por GNC dejó de invocarse espuriamente en los 6 casos. El de
granizo pasó de un rechazo liso a "requiere peritaje", que es defendible: el
manual **es ambiguo** ahí, porque la lista general de pág. 10 incluye GNC pero
con la salvedad del "excepto".

### Qué queda sin resolver

El **falso positivo del robo**: el sistema cubre un robo sobre un vehículo con
GNC, que las listas de pág. 10 y pág. 16 excluyen.

**Causa raíz identificada:** no es un problema de razonamiento sino de
**recuperación**. Para la consulta *"Me robaron el auto AF987XX"* ninguno de los
12 chunks recuperados contiene la exclusión por GNC, así que el LLM nunca la ve.

El motivo está en `nodo_consultar_clausulas` (`agent.py`): la búsqueda usa
**sólo el texto del usuario**, que no menciona el GNC. El dato "este vehículo
tiene GNC" existe en SQL (`vehiculos.tiene_gnc`) pero nunca entra a la consulta.

### Próximo paso sugerido

Enriquecer la consulta al RAG con los atributos de la póliza antes de buscar:

```python
def nodo_consultar_clausulas(state):
    mensaje = state["messages"][-1].content
    # Sumar a la consulta los atributos que disparan exclusiones (GNC, uso,
    # producto), para que se recuperen las cláusulas que dependen de ellos y
    # no sólo las que el usuario nombró.
    consulta = f"{mensaje} {atributos_relevantes(state['datos_poliza'])}"
    ...
```

Alternativa complementaria: una segunda búsqueda dedicada a exclusiones, cuyos
resultados se agreguen al contexto del dictamen.

Cómo verificar el arreglo: correr la batería de 6 casos y confirmar que
*robo + GNC* pasa a **Rechazado** sin que se rompan los otros cinco.
