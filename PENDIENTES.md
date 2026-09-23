# Informe consolidado — San Cris (TP2)

Estado del sistema frente a los **requisitos de cátedra**, detalle de faltantes y
**plan de implementación**. Documento único (reemplaza a `REQUISITOS.md`).

**Fuentes de los requisitos** (`notebooks_clases/`):
- `nlp_1_conceptos (1).py` → conceptos NLP, mini-RAG, agente de juguete **con traza**.
- `nlp_3_langchain_conversaciones_rag_v2 (2).py` → LCEL, memoria, RAG, **history-aware**, RAG con LangGraph.
- `copia_de_nlp_4_langchain_agentes (1).py` → **agente moderno `create_agent`**, tools, backend intercambiable, **traza observable**.

**Leyenda:** ✅ Cumple · ⚠️ Parcial · ❌ Falta · — No aplica

---

## 1. Estado frente a los requisitos de cátedra

### Grupo A — Agente moderno (`nlp_4`)

| ID | Requisito | Estado |
|---|---|---|
| A1 | `create_agent` (LangChain 1.x sobre LangGraph): **el modelo decide** responder o llamar tools (`tool_calls`). Patrón ReAct. | ❌ |
| A2 | Tools con `@tool`: descripción desde docstring, esquema desde tipos. | ✅ |
| A3 | Tools verificadas localmente **sin LLM** (asserts). | ⚠️ |
| A4 | Backend **separado**: factory `make_llm(backend)`, modelo intercambiable. | ❌ |
| A5 | `SYSTEM_PROMPT` con rol y reglas. | ✅ |
| A6 | Memoria con checkpointer + `thread_id`. | ✅ |
| A7 | **Traza observable**: herramienta, argumentos, resultado, respuesta. | ❌ |
| A8 | `recursion_limit` en el invoke. | ⚠️ |
| A9 | Modelo con soporte de tool calling. | ⚠️ |

### Grupo B — Memoria y conversación (`nlp_3`)

| ID | Requisito | Estado |
|---|---|---|
| B1 | `RunnableWithMessageHistory` + `session_id`. | ⚠️ (cubierto por checkpointer, §3.5) |
| B2 | Resumen / recorte / selección del historial para conversaciones largas. | ❌ |
| B3 | Persistencia del historial entre sesiones (FileChatMessageHistory o DB). | ❌ |
| B4 | RAG conversacional con LangGraph (`StateGraph`, `add_messages`, `get_state`). | ✅ |

### Grupo C — Pipeline RAG (`nlp_3` §3)

| ID | Requisito | Estado |
|---|---|---|
| C1 | Carga de documentos (`PyPDFLoader`/`DirectoryLoader`). | ✅ |
| C2 | Chunking con `RecursiveCharacterTextSplitter` ajustable. | ✅ (superado: estructural) |
| C3 | Embeddings + **Chroma** persistente. | ✅ |
| C4 | Retriever con `k` configurable. | ✅ (superado: híbrido BM25+vector) |
| C5 | Cadena **history-aware**: reformular el seguimiento antes de recuperar. | ❌ |
| C6 | Mostrar **fuentes** / inspeccionar el vector store (`similarity_search_with_score`). | ❌ |
| C7 | Reintentos con backoff ante errores temporales. | ✅ |

### Grupo D — LCEL y conceptos (`nlp_1`, `nlp_3` §1)

| ID | Requisito | Estado |
|---|---|---|
| D1 | LCEL (`prompt \| modelo \| parser`). | ⚠️ (opcional; `nlp_4` no lo usa) |
| D2 | Buenas prácticas: revisar *deprecations* y actualizar imports. | ⚠️ |
| D3 | Tokenización/lematización NLTK + TF-IDF/Naive Bayes. | — (concepto, no requisito del agente) |
| D4 | Agente de juguete que decide herramienta y expone su traza. | ❌ (igual que A7) |

**Resumen:** 9 cumplidos · 5 parciales · 8 faltantes (varios se resuelven en bloque).

---

## 2. Detalle de faltantes

### A1 + A7 — Patrón `create_agent` / ReAct y traza observable  · CRÍTICO
Hoy el flujo es un **grafo determinístico**: el LLM solo clasifica intención
(`nodo_clasificar`) y el ruteo es por reglas (`rutear_desde_clasificar`). Las
tools se invocan como funciones comunes, no como herramientas de un agente, y no
hay traza de `tool_calls`. El notebook `nlp_4` es explícito: el modelo debe
decidir qué herramienta usar y la traza (herramienta, argumentos, resultado)
debe ser observable.

**Resolución:** enfoque **híbrido**. Mantener determinístico el camino crítico
(`consultar_poliza` → `consultar_clausulas` → `emitir_dictamen` → `registrar`)
por auditabilidad, y agregar un subgrafo ReAct con tools bindeadas para la parte
abierta de consulta. **Dónde:** `agent.py`, `tools.py`.

### A4 — Backend intercambiable  · CRÍTICO
`llm = ChatGroq(...)` está hardcodeado en `agent.py`. El notebook dice textual:
*"El backend se mantiene separado del resto del agente para mostrar que el modelo
es un componente intercambiable."*

**Resolución:** módulo `llm.py` con factory `make_llm(backend)`, backend leído
desde `config.py`/`.env`. **Dónde:** `llm.py` (nuevo), `config.py`, `agent.py`.

### C5 — Cadena history-aware  · ALTA
`nodo_consultar_clausulas` y `nodo_clasificar` leen **solo** el último mensaje.
Un seguimiento ("¿y con granizo?") llega suelto al retrieval. `nlp_3` §3.4 lo
resuelve con un `contextualize_q_prompt` que reformula usando el historial.

**Resolución:** nodo `condensar` previo a clasificar/recuperar. **Dónde:** `agent.py`.

### B2 — Recorte / resumen del historial  · ALTA
`nodo_responder` hace `mensajes.extend(state["messages"])`: historial completo,
siempre. Con el free tier de Groq (200K tokens/día **compartidos**), una charla
larga agota el cupo y mete ruido. `nlp_3` §2 lo pide explícitamente.

**Resolución:** `trim_messages` (ventana) + resumen incremental por umbral.
**Dónde:** `agent.py`, `config.py`.

### B3 — Persistencia entre sesiones  · MEDIA
`MemorySaver` vive en RAM: cerrar la app pierde la conversación.

**Resolución:** `SqliteSaver` (`langgraph-checkpoint-sqlite`). **No** construir
sobre `FileChatMessageHistory` (vive en `langchain_community`, en sunset).
**Dónde:** `agent.py`, `config.py`, `app.py`.

### C6 — Fuentes / evidencia del RAG  · MEDIA
`buscar_clausulas` devuelve texto y **descarta** metadata y score de los chunks.

**Resolución:** función que devuelva texto + evidencia
(`{seccion, pagina, fuente, score, preview}`); `buscar_clausulas` queda de wrapper.
**Dónde:** `rag.py`.

### A3 — Tools verificadas sin LLM  · BAJA
`test_casos.py` prueba vía el agente (necesita LLM). `nlp_4` valida tools con
asserts locales. **Resolución:** `test_tools.py` sin LLM. **Dónde:** `test_tools.py` (nuevo).

### A8 — `recursion_limit`  · BAJA
No seteado. **Resolución:** agregarlo a la config del invoke. **Dónde:** `main.py`, `app.py`.

### B1 — Tuning de chunking  · BAJA
`CHUNK_SIZE=1000` / `CHUNK_OVERLAP=300` = 30% de solapamiento (alto). Con `k=6`,
infla el contexto. **Resolución:** medir 800/150 vs 1000/200. **Dónde:** `config.py`.

### D2 — Deprecations e imports  · BAJA
`langchain-community` en sunset. Migrar Chroma → `langchain_chroma`. Modernizar
`MemorySaver` → `InMemorySaver`. **Dónde:** `rag.py`, `agent.py`, `requirements.txt`.

### D1 — Demo LCEL  · OPCIONAL
`nlp_3` §1 enseña LCEL; `nlp_4` no lo usa. **Resolución:** cadena LCEL mínima como
demo en el notebook de entrega. **Dónde:** notebook.

---

## 3. Plan de implementación

Orden por dependencias e impacto. Cada fase cierra requisitos y es verificable
de forma independiente. Las fases 1 y 2 son el núcleo para la nota.

### Fase 0 — Higiene y base  *(rápida, sin riesgo)*
- **Cierra:** A8, A3, D2, B1, A6 (modernización).
- **Pasos:** agregar `recursion_limit`; crear `test_tools.py` con asserts sin LLM;
  migrar a `langchain_chroma`; `MemorySaver` → `InMemorySaver`; experimento de
  chunk/overlap.
- **Verificación:** `test_tools.py` pasa sin API key; sin warnings de sunset.

### Fase 1 — Backend intercambiable  *(prerequisito de Fase 2)*
- **Cierra:** A4.
- **Pasos:** crear `llm.py` con `make_llm(backend: Literal[...])`; agregar
  `LLM_BACKEND` y modelos por backend a `config.py`; reemplazar la instancia
  hardcodeada en `agent.py`. Groq queda como un backend más.
- **Verificación:** cambiar proveedor solo por `.env`, sin tocar `agent.py`.

### Fase 2 — Agente ReAct híbrido + traza  *(núcleo de la nota)*
- **Cierra:** A1, A7, D4, A9 (y parte de C6).
- **Pasos:**
  1. Construir un subagente con `create_agent` (`langchain.agents`) con las tools
     bindeadas (`consultar_datos_poliza`, `consultar_clausulas_poliza`,
     `consultar_polizas_por_dni`, `consultar_estado_reclamo`,
     `consultar_siniestros_previos`) + `SYSTEM_PROMPT`.
  2. Insertarlo como nodo/subgrafo para la parte abierta; mantener el camino
     determinístico para `siniestro`.
  3. Capturar la traza de `result["messages"]`: `tool_calls` (decisión) y
     mensajes `type=="tool"` (observación).
- **Verificación:** una consulta general muestra al modelo eligiendo la tool;
  el dictamen de siniestro sigue siendo determinístico.

### Fase 3 — History-aware + recorte/resumen
- **Cierra:** C5, B2.
- **Pasos:** nodo `condensar` (contextualize) antes de clasificar/recuperar;
  `trim_messages` + resumen por umbral configurable en `responder`.
- **Verificación:** "¿y con granizo?" recupera cláusulas correctas; el historial
  largo no crece sin límite.

### Fase 4 — Persistencia
- **Cierra:** B3, B1.
- **Pasos:** `SqliteSaver` con path en `config.py`; `thread_id` por sesión en
  `app.py`.
- **Verificación:** cerrar y reabrir la app conserva la conversación.

### Fase 5 — UI de trazabilidad y fuentes
- **Cierra:** A7 (UI), C6, M2.
- **Pasos:** `rag.py` devuelve evidencia; campo `evidencia` en el estado; en
  `app.py` un `st.expander("Cómo llegué a esto")` por turno con pestañas
  Razonamiento / Tools / Chunks + traza del último turno en el sidebar.
- **Verificación:** cada respuesta expone tools llamadas y chunks con score.

### Fase 6 — Documentación y entrega
- **Cierra:** D1 (demo LCEL), entregable del TP.
- **Pasos:** actualizar `DOCUMENTACION.md` (decisiones y dificultades nuevas);
  empaquetar el sistema en el **notebook interactivo** de entrega, incluyendo la
  traza y una cadena LCEL mínima como demo.
- **Verificación:** el notebook corre de punta a punta en Colab.

**Dependencias:** Fase 1 → Fase 2. Fases 3, 4 y 5 son independientes entre sí
(se pueden paralelizar tras la Fase 2). Fase 0 puede ir en cualquier momento.
