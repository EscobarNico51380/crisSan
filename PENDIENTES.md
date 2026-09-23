# Informe de faltantes y mejoras — San Cris (TP2)

Estado del proyecto frente a los contenidos de la cátedra y buenas prácticas de
LangChain / LangGraph. Ordenado por severidad. Cada ítem indica **qué falta**,
**por qué importa** y **dónde tocar**.

---

## CRÍTICO

### C1. La selección de herramientas no es ReAct: hoy es un grafo determinístico

**Qué falta.** En el flujo actual el LLM **no decide qué herramienta usar**. El
único punto donde interviene el modelo para el ruteo es `nodo_clasificar`
(`agent.py:275`), que clasifica la intención (`consulta` / `siniestro` /
`general` / `charla`). A partir de ahí el camino es 100% reglas fijas
(`rutear_desde_clasificar`, `agent.py:455`) y las tools se invocan como
**funciones Python comunes** (`consultar_datos_poliza.invoke(...)`,
`agent.py:350`), no como herramientas de un agente.

**Por qué importa.** El enunciado del TP describe explícitamente el patrón
ReAct: *"el LLM alterna entre razonar una pregunta y decidir utilizar una
herramienta específica"* (`IdeaTP.md`, líneas 36-37). El criterio de evaluación
**"Aplicación del marco teórico"** espera ver esa decisión del modelo. Con el
diseño actual, la defensa oral no puede mostrar al LLM eligiendo tools: solo
puede mostrar clasificación de intención + ruteo por reglas.

**Cómo resolverlo — enfoque HÍBRIDO (recomendado), no ReAct puro.**
Un ReAct libre sobre TODO el flujo es un retroceso para este caso: el dictamen
de cobertura y el registro del siniestro exigen determinismo y auditabilidad
(hoy garantizados por el grafo). El diseño correcto combina lo mejor de ambos:

- **Camino crítico determinístico (se mantiene):** `consultar_poliza` →
  `consultar_clausulas` → `emitir_dictamen` → `registrar`. Nadie quiere que un
  LLM improvise una decisión de cobertura ni que registre dos veces un
  siniestro.
- **Loop ReAct para la parte abierta:** un agente con las tools bindeadas
  (`create_react_agent` de `langgraph.prebuilt`, o `create_agent` de
  `langchain.agents` — ambos disponibles en el entorno) que resuelva las
  preguntas de consulta/RAG eligiendo herramientas por sí mismo.

Con esto el LLM **sí** selecciona herramientas (cumple el marco teórico) sin
perder la confiabilidad del dictamen.

**Dónde tocar.** `agent.py` (nuevo nodo/subgrafo ReAct + bind de tools),
`tools.py` (ya están decoradas con `@tool`, solo falta entregárselas a un
agente). Registrar la decisión actual ("grafo determinístico") como limitación
conocida en `DOCUMENTACION.md` §§ 9 y 13.

---

## ALTA — Calidad de respuesta

### A1. Preguntas de seguimiento: falta el patrón *history-aware*

`nodo_consultar_clausulas` (`agent.py:355`) y `nodo_clasificar` (`agent.py:265`)
leen **solo** `state["messages"][-1].content`. Un seguimiento como "¿y con
granizo?" llega al retrieval como frase suelta: se recuperan cláusulas al azar
y se clasifica mal.

**Fix.** Nodo `condensar` previo a clasificar/recuperar: un LLM reescribe el
seguimiento a pregunta autónoma usando el historial. Es el patrón *history-aware
retriever* de LangChain. La persistencia de `patente`/`dni`/`reclamo` en estado
resuelve los datos, pero **no** el texto de la consulta al RAG.

### A2. Historial sin recorte ni resumen: el contexto crece indefinidamente

`nodo_responder` (`agent.py:438`) hace `mensajes.extend(state["messages"])`: el
historial completo, siempre. Sin ventana, sin recorte, sin resumen. Con el free
tier de Groq (200K tokens/día **compartidos por organización**), una charla
larga agota el cupo y degrada la respuesta por ruido acumulado.

**Fix.** `trim_messages` (`langchain_core.messages`) para ventana deslizante +
resumen incremental cuando el historial supera un umbral.

### A3. El modelo no es un componente intercambiable

`llm = ChatGroq(...)` está instanciado en el import de `agent.py:47`,
hardcodeado. La cátedra pide el backend separado del agente.

**Fix.** Módulo `llm.py` con factory `get_llm()` que lea el proveedor desde
`config.py` / `.env`. Cambiar Groq ↔ OpenAI ↔ Ollama pasa a ser una variable de
entorno.

---

## MEDIA

### M1. Persistencia del historial entre sesiones

`MemorySaver` (`agent.py:530`) vive en RAM: al cerrar Streamlit o la consola se
pierde toda la conversación.

**Fix.** `SqliteSaver` de `langgraph-checkpoint-sqlite` (requiere
`pip install langgraph-checkpoint-sqlite`), apuntando a un SQLite. Mantiene
LangGraph y cumple el punto de la cátedra ("o una base de datos").
**Nota:** NO construir sobre `FileChatMessageHistory` — vive en
`langchain_community`, que está oficialmente en sunset (avisa en runtime).

### M2. Trazabilidad: UI de razonamiento, tools y chunks

No se puede ver **de dónde** sale cada respuesta. `buscar_clausulas`
(`rag.py:203`) devuelve texto formateado y **descarta** la metadata de los
chunks (sección, página, score).

**Fix (3 piezas):**
1. `rag.py`: función que devuelva texto + evidencia
   (`{seccion, pagina, fuente, score, preview}`); `buscar_clausulas` queda como
   wrapper.
2. `agent.py`: campo `evidencia` en el estado (reducer `operator.add`) que cada
   nodo alimenta (clasificación, ruteo, tool, chunks, dictamen).
3. `app.py`: `st.expander("Cómo llegué a esto")` por turno con pestañas
   Razonamiento / Tools / Chunks; traza del último turno en el sidebar.

---

## BAJA — Higiene técnica

### B1. Tuning de chunking y retrieval

`CHUNK_SIZE=1000` / `CHUNK_OVERLAP=300` (`config.py:42-43`) = **30% de
solapamiento**, alto: chunks contiguos comparten casi un tercio del texto, el
retriever devuelve casi lo mismo dos veces e infla el contexto. Con `k=6`
(`rag.py:203`) y chunks grandes, el problema es real.

**Fix.** Medir con 800/150 o 1000/200 y comparar calidad de recuperación.

### B2. Deprecations y migración de imports

`langchain-community` está en sunset (lo avisa en runtime). Migrar
`langchain_community.vectorstores.Chroma` → `langchain_chroma`
(`pip install langchain-chroma`).

**Nota sobre `RunnableWithMessageHistory`:** la cátedra lo menciona como "el
enfoque moderno", pero con LangChain 1.x es API legacy. El camino moderno es el
**checkpointer de LangGraph**, que el proyecto ya usa. En arquitectura de
memoria el proyecto está alineado; RWMH solo si el docente lo pide
explícitamente (mostrarlo como demo de LCEL, no como reemplazo).

### B3. Nota sobre `EnsembleRetriever`

Verificado: en `langchain` 1.4.2 **no existe** `langchain.retrievers.EnsembleRetriever`;
la ubicación correcta es `langchain_classic.retrievers` (la que ya usa el
proyecto). No hay nada que cambiar acá.
