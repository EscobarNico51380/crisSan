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

**Estado** (`EstadoAgente`): `messages`, `intencion`, `consulta`, `patente`, `dni`,
`reclamo`, `datos_poliza`, `clausulas`, `exclusiones`, `dictamen`,
`siniestro_registrado`, `reclamo_resultado`.

**Nodos:**

| Nodo | Qué hace |
|---|---|
| `clasificar` | Extrae patente/DNI con **regex** + clasifica intención con LLM (`json_schema`). Ve los últimos mensajes y devuelve además la pregunta reescrita para que se entienda sola, que es la que se usa para buscar (§15) |
| `pedir_patente` | Pide la patente si falta |
| `listar_polizas` | Lista las pólizas de un DNI y pide elegir |
| `consultar_reclamo` | Consulta el estado de un reclamo por su número |
| `consultar_poliza` | Llama a la tool de SQL |
| `consultar_clausulas` | RAG híbrido sobre el manual; si el vehículo tiene GNC, además busca las exclusiones que dependen de eso (§14) |
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
| 23 | Robo e incendio de un auto con GNC salían cubiertos, aunque el manual los excluye | La búsqueda usaba solo el texto del usuario, que no menciona el GNC: la exclusión nunca se recuperaba | Segunda búsqueda, dirigida a exclusiones, cuando el vehículo tiene GNC según la base (§14) |
| 24 | Las preguntas de seguimiento ("¿y cuándo vence?") se clasificaban y buscaban sueltas | El clasificador y la búsqueda leían solo el último mensaje | El clasificador recibe los últimos mensajes y devuelve la pregunta reescrita, en la misma llamada al LLM (§15) |

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
- Las exclusiones que dependen del vehículo solo contemplan el **GNC** (ver §14).

---

## 14. El bug del GNC: diagnóstico y arreglo

### El problema

El sistema daba por cubiertos siniestros que el manual excluye para vehículos
adaptados a GNC. El manual tiene **tres** listas que mencionan esa exclusión, y
cada una tiene un alcance distinto:

| Página | Alcance de la exclusión |
|---|---|
| 10 | General — *"excepto que la cobertura incluya alguno de los puntos enumerados"* |
| 15 | Sólo **Incendio** |
| 16 | Sólo **Robo o Hurto** |

### Cómo se probó

Seis siniestros, cada uno en una conversación nueva, cruzando el tipo de
siniestro con los dos autos de prueba: el Peugeot 208 (`AF987XX`, producto C+,
**con GNC**) y el Ford Focus (`AB123CD`, producto D, sin GNC). "Según el
manual" es la respuesta que corresponde leyendo las cláusulas. Antes de cada
corrida se vació la tabla `siniestros`, para que los dictámenes de corridas
anteriores no influyeran como precedentes.

| Pregunta literal | Según el manual | Original | Etapa 1 | Etapa 2 |
|---|---|---|---|---|
| *Me cayo granizo sobre la AF987XX* | **Ambiguo** | Rechazado | Requiere peritaje | Cubierto |
| *Me rompieron el parabrisas de la AF987XX* | **Ambiguo** | Cubierto | Cubierto | Cubierto |
| *Se incendio por completo la AF987XX* | **Rechazado** (pág. 15) | Cubierto ❌ | Cubierto ❌ | Rechazado ✅ |
| *Me robaron el auto AF987XX* | **Rechazado** (pág. 16) | Cubierto ❌ | Cubierto ❌ | Rechazado ✅ |
| *Me cayo granizo sobre la AB123CD* | Cubierto | Cubierto ✅ | Cubierto ✅ | Cubierto ✅ |
| *Me robaron el auto AB123CD* | Cubierto | Cubierto ✅ | Cubierto ✅ | Cubierto ✅ |

**Los casos ambiguos.** La lista general de pág. 10 excluye los vehículos con
GNC, pero con la salvedad *"excepto que la cobertura incluya alguno de los
puntos enumerados"*. Las listas de pág. 15 y 16 repiten la exclusión solo para
incendio y robo, lo que sugiere que ahí está su alcance real. Para granizo y
cristales el manual no es concluyente: tanto "Cubierto" como "Requiere
peritaje" son defendibles.

> **Corrección.** En la primera versión de esta batería el incendio figuraba
> como esperado "Cubierto" y, por lo tanto, como caso correcto. Releyendo el
> manual, la lista *"Exclusiones a la cobertura para Incendio"* de pág. 15
> incluye el GNC: lo correcto es **Rechazado**. Era un segundo falso positivo,
> con la misma causa que el robo.

### Etapa 1 — Los fragmentos perdían su encabezado (commit `148f605`)

**Causa.** El chunking partía el manual por ítems numerados, y cada ítem
quedaba sin el título que le da alcance. El fragmento recuperado decía solo
*"5) Cuando el vehículo haya sido adaptado para ser propulsado por GNC"*, sin
indicar que pertenece a *Exclusiones a la cobertura para Robo o Hurto*. El LLM
lo leía como una exclusión absoluta.

**Arreglo.** Al indexar, cada ítem (numerado o con viñeta) lleva adelante el
encabezado de su lista.

**Resultado.** La exclusión dejó de aplicarse donde no correspondía: el granizo
dejó de rechazarse. Pero el robo y el incendio seguían saliendo cubiertos.

### Etapa 2 — La exclusión nunca se recuperaba

**Causa.** No era un problema de razonamiento sino de **recuperación**. La
búsqueda en el manual usaba solo el texto del usuario, y nadie escribe "tengo
GNC" al denunciar un robo. Para *"Me robaron el auto AF987XX"*, ninguno de los
12 fragmentos recuperados contenía la exclusión, así que el LLM nunca la veía.
El dato "este vehículo tiene GNC" estaba en SQL (`vehiculos.tiene_gnc`), pero
no se usaba para buscar.

**Arreglo.** En `nodo_consultar_clausulas`, si la intención es una consulta o
un siniestro y el vehículo tiene GNC, se hace una **segunda búsqueda**:

1. Con una consulta armada por el sistema, no por el usuario
   (`CONSULTA_EXCLUSIONES_GNC`).
2. Quedándose solo con los fragmentos que son exclusiones y mencionan el GNC
   (`rag.buscar_exclusiones`). Así queda afuera, por ejemplo, el texto que
   obliga a *declarar* el equipo de GNC, que no es una exclusión.
3. El resultado entra al prompt del dictamen en una sección propia, *EXCLUSIONES
   POR CARACTERÍSTICAS DEL VEHÍCULO*, y el prompt suma una regla: cada exclusión
   se aplica solo a la cobertura que indica su encabezado.

**Por qué una búsqueda aparte** y no agregar "GNC" a la pregunta del usuario:
los fragmentos sobre robo y los de GNC competirían por los mismos 12 lugares y
se desplazarían entre sí. Separados, cada búsqueda trae lo suyo, y en el prompt
queda claro qué es el relato y qué son las exclusiones.

**Resultado.** 6 de 6. El robo y el incendio pasaron a *Rechazado*. El granizo
y el parabrisas siguen *Cubierto* aunque ahora el modelo **ve** las
exclusiones: las lee y entiende que no aplican a esas coberturas.

**Costo.** Una búsqueda extra, local (no consume API), y solo para vehículos
con GNC.

### Limitaciones

- Solo se contempla el **GNC**, que es la característica que se verificó. El
  manual también excluye vehículos adaptados a gas licuado de petróleo, pero la
  base de datos no registra ese dato. Otra característica se sumaría con el
  mismo mecanismo: su consulta, sus palabras clave y el dato en la base.
- En los casos ambiguos (granizo, cristales) el sistema hoy responde "Cubierto".
  Si el grupo decide que lo correcto es "Requiere peritaje", hay que ajustar el
  prompt del dictamen.


---

## 15. Preguntas de seguimiento

### El problema

El clasificador y la búsqueda en el manual leían **solo el último mensaje**.
Una pregunta que depende de la conversación llegaba suelta:

- *"¿Y cuándo vence?"*, después de consultar una póliza, se clasificaba como
  pregunta **general** sobre el manual, no como consulta de la póliza. La
  respuesta salía bien por casualidad: la fecha estaba en el historial.
- *"¿Y me cubre el granizo?"* buscaba en el manual sin saber de qué póliza se
  hablaba.

### La solución

El clasificador ya llamaba al LLM en cada turno. Se aprovecha esa misma
llamada: ahora recibe los **últimos 4 mensajes** (cada uno recortado a 300
caracteres) y, además de la intención, devuelve la pregunta **reescrita para
que se entienda sola** (campo `consulta_autonoma` del schema, `consulta` en el
estado). Esa versión es la que se usa para buscar en el manual. En la base de
datos se sigue guardando el mensaje original del usuario.

Se descartaron dos alternativas:

| Alternativa | Por qué no |
|---|---|
| Una llamada extra al LLM, solo para reescribir, en todos los mensajes | Suma demora y cupo en cada turno |
| Reescribir solo si el mensaje "parece" de seguimiento (corto, empieza con "¿y...") | Una regla fija se equivoca con frases que no previó |

**Costo:** ninguna llamada extra al LLM, solo unos cientos de tokens más de
entrada por turno. En el primer mensaje de una conversación no hay historial y
la entrada queda igual que antes.

### Un riesgo que hubo que cubrir

Con el historial a la vista, una **hipótesis** después de un siniestro (*"¿y
si hubiera sido granizo?"*) podía clasificarse como un siniestro nuevo, y el
sistema habría **registrado un reclamo que nunca ocurrió**. El prompt del
clasificador ahora aclara que "siniestro" es solo un hecho que ocurrió; una
hipótesis es una consulta.

### Resultados

| Conversación | Intención | Pregunta reescrita | Resultado |
|---|---|---|---|
| *¿Qué plan tengo para la patente AB123CD?* → *Y cuando vence?* | `consulta` (antes: `general`) | *¿Cuándo vence la póliza de la patente AB123CD?* | ✅ |
| *¿Qué plan tengo para la patente AF987XX?* → *y me cubre el granizo?* | `consulta` | *¿Me cubre el granizo la póliza de la patente AF987XX?* | ✅ cita la cláusula del granizo |
| *Choque contra un poste con la AF987XX* → *y si hubiera sido granizo?* | `consulta` | *¿Estaría cubierto mi vehículo si el daño hubiera sido causado por granizo?* | ✅ no registró ningún reclamo |

Se repitió además la batería del GNC (§14) para verificar que el clasificador
nuevo no cambiara los dictámenes.
