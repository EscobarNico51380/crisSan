# Pendientes — San Cris (TP2)

Una sola lista del grupo. Cuando algo se termina, se marca ✅ y se anota el
commit. Al final está el estado del sistema frente a lo que piden los notebooks
de la cátedra.

**Fechas:** video de 10-15 minutos el **30/09** (todos los grupos). Defensa
oral el 30/09 o el 14/10 (turno tarde/noche), o el 01/10 o el 08/10 (turno
mañana); incluye 1-2 preguntas sobre el TP1.

**Formato de entrega (confirmado por el profesor):** proyecto de varios `.py`
con la documentación, siempre que incluya las instrucciones para reproducirlo.
No hace falta notebook.

---

## Problemas (cosas que andaban mal)

| Problema | Estado | Commit |
|---|---|---|
| Streamlit no abría en Windows (el ícono de la página) | ✅ | `a0bf355` |
| El entorno no era reproducible (versiones sin fijar) | ✅ | `0198a99` |
| `torchvision` hacía falta en Linux y no en Windows | ✅ Se instala solo en Linux | `20191bb` |
| Los precedentes eran los últimos casos, no casos comparables | ✅ | `361026d` |
| La exclusión por GNC se aplicaba al granizo | ✅ | `148f605` |
| El robo y el incendio de un auto con GNC salían cubiertos | ✅ | `5363974` |
| Las preguntas de seguimiento ("¿y cuándo vence?") se interpretaban sueltas | ✅ | `881329f` |
| El historial que se manda al LLM crecía sin límite | ✅ Ventana + resumen | `d5e8091` |
| Al saludar, responde pidiendo la patente | ✅ | `7e42d4f` |
| Al agotarse el cupo, el chat mostraba el error técnico crudo | ✅ Mensaje amigable con la espera | `333af5d` |
| Una póliza vencida sale cubierta | ✅ La vigencia y el pago se controlan con código, sin el LLM | `a48070e` |
| "Me robaron el auto AJ222NP el 10 de mayo de 2026" se clasificó como consulta y no como denuncia | ⏳ Visto con el modelo chico | Revisar con el principal |
| La clase `Chroma` que usamos está deprecada | Se deja así: el proyecto dura un mes | |

## Features

En orden sugerido (el plan de Nicolás): primero poder cambiar de modelo, porque
el agente ReAct y las imágenes se apoyan en eso.

| Feature | Estado | Notas |
|---|---|---|
| Costo estimado de cada consulta, visible en el chat | ✅ | `4906c02` |
| Recorrido sugerido en la web: botones con las consultas recomendadas | ✅ | `2d828c8` |
| Más autos de prueba en la base | ✅ 6 autos nuevos, cada uno prueba una sola cosa | `a7f8f3f` |
| Correr con el modelo grande los 5 casos de autos nuevos que se cortaron por el cupo | ⏳ | Con el modelo chico ya pasaron 4 de 5; falló la póliza vencida (ver Problemas) |
| Modelo de respaldo automático cuando el principal agota su cupo, sin reintentar el agotado | ✅ | `c72e8af` |
| Poder cambiar de modelo desde la configuración (`make_llm`, como en el notebook de agentes) | 🔶 Casi | `llm.py` ya crea los modelos con `make_llm` y los nombres están en `config.py`. Falta poder elegir otro proveedor (OpenAI, Ollama) |
| Que el LLM elija las herramientas (patrón ReAct con `create_agent`), dejando fijo el camino del dictamen | ⏳ | Lo que más suma en "aplicación del marco teórico". Falta decidir qué preguntas pasan por ahí |
| Mostrar en la web de dónde sale cada respuesta: herramienta elegida, con qué datos, qué devolvió, y los fragmentos del manual con su puntaje | ⏳ | Es la "traza observable" del notebook de agentes. Luce mucho en el video |
| Interpretar imágenes enviadas por el cliente | ⏳ | Prometido en el mail al profesor. Modelo verificado: `qwen/qwen3.8-27b` (~1.300-1.800 tokens por foto). Falta decidir si describe la foto para el dictamen actual o si dictamina él |
| Exclusiones por gas licuado (GLP) y por uso para transporte de pasajeros | ⏳ A confirmar | El manual las tiene; se sumarían con el mismo mecanismo que el GNC |
| Historial que sobreviva al cerrar la app (`SqliteSaver`) | ⏳ Baja prioridad | Suma una dependencia |
| Pruebas de las herramientas sin usar el LLM | ⏳ Baja prioridad | Como las del notebook de agentes; hoy `test_casos.py` pasa siempre por el LLM |
| Límite de recursión en cada consulta (`recursion_limit`) | ⏳ Baja prioridad | Importa cuando exista el agente ReAct, que puede entrar en bucle |
| Cambiar `MemorySaver` por `InMemorySaver` | ⏳ Baja prioridad | Nombre moderno de lo mismo |
| Ajustar el tamaño y el solapamiento de los fragmentos del manual | ⏳ Baja prioridad | Cualquier cambio obliga a repetir la prueba del GNC |
| Demo mínima de LCEL (`prompt \| modelo \| parser`) | ⏳ Opcional | La enseña el notebook de RAG; el sistema no la necesita |

## Por decidir

- **Granizo o parabrisas en un auto con GNC.** El manual es ambiguo: la lista
  general de exclusiones incluye el GNC, pero con la salvedad "excepto que la
  cobertura incluya alguno de los puntos enumerados". Hoy el sistema responde
  "Cubierto"; la otra lectura defendible es "Requiere peritaje". Tiene que
  coincidir con lo que se diga en la defensa. **Además hay que escribirlo como
  regla en el prompt:** con el modelo chico el mismo caso sale "Rechazado", así
  que hoy la respuesta depende del modelo.
  **Decisión provisoria:** queda documentado y se retoma más adelante. Si leyendo
  la póliza entera no se puede resolver, la respuesta será "Requiere peritaje".

## Entregables

| Entregable | Estado |
|---|---|
| Slides de la defensa | ⏳ |
| Guion y grabación del video (vence el 30/09) | ⏳ |
| Vaciar la tabla de siniestros antes de grabar (las pruebas quedan como precedentes) | ⏳ |
| Repaso del TP1 | ⏳ |

## Cierre (lo último antes de entregar)

| Tarea | Estado |
|---|---|
| `README.md` con los integrantes, y `iniciar.py`: instala y abre el chat con un solo comando | ✅ | (este commit) |
| Decidir qué hacer con `ONBOARDING.md` y `SETUP.md`, ahora que el README cubre lo que necesita el profesor | ⏳ |
| Armar el zip de la entrega: sin `.venv` ni `data/modelos`, con el índice ya armado y el `.env` con una key de Groq de una cuenta nueva dedicada a la entrega (1-2 meses de vida, revocarla después). Aclarar en el README que la key está incluida | ⏳ |
| Probar la instalación desde cero con el zip, siguiendo solo el README, en Windows y en Linux | ⏳ |

---

## Estado frente a los notebooks de la cátedra

Análisis de Nicolás, con los estados actualizados. Los notebooks, pasados a
`.py`, están en `notebooks_clases/`.

**Leyenda:** ✅ cumple · ⚠️ parcial · ❌ falta · — no aplica

### Agentes (`copia_de_nlp_4_langchain_agentes`)

| Requisito | Estado |
|---|---|
| `create_agent`: el modelo decide si responder o llamar una herramienta (ReAct) | ❌ |
| Herramientas con `@tool` (descripción desde el docstring, esquema desde los tipos) | ✅ |
| Herramientas verificadas localmente, sin LLM | ⚠️ Se prueban a través del agente |
| Backend separado, modelo intercambiable (`make_llm`) | ❌ |
| Prompt de sistema con rol y reglas | ✅ |
| Memoria con checkpointer + `thread_id` | ✅ |
| Traza observable: herramienta, argumentos, resultado | ❌ |
| `recursion_limit` en la consulta | ⚠️ |
| Modelo con soporte de *tool calling* | ⚠️ `gpt-oss-120b` lo soporta, pero hoy no se usa |

### Memoria y RAG (`nlp_3_langchain_conversaciones_rag_v2`)

| Requisito | Estado |
|---|---|
| `RunnableWithMessageHistory` + `session_id` | ⚠️ Cubierto por el checkpointer de LangGraph, que es el enfoque moderno |
| Resumen o recorte del historial en conversaciones largas | ✅ Ventana + resumen (`d5e8091`) |
| Historial persistente entre sesiones | ❌ |
| RAG conversacional con LangGraph (`StateGraph`, `add_messages`, `get_state`) | ✅ |
| Carga de documentos (`PyPDFLoader`) | ✅ |
| Chunking con `RecursiveCharacterTextSplitter` | ✅ Mejorado: corte por la estructura del manual |
| Embeddings + Chroma persistente | ✅ |
| Retriever con `k` configurable | ✅ Mejorado: búsqueda híbrida (BM25 + semántica) |
| Pregunta reformulada con el historial antes de buscar (*history-aware*) | ✅ Lo hace el clasificador en la misma llamada (`881329f`) |
| Mostrar las fuentes y los puntajes de la búsqueda | ❌ |
| Reintentos con espera creciente ante errores temporales | ✅ |

### Conceptos (`nlp_1_conceptos`)

| Requisito | Estado |
|---|---|
| LCEL (`prompt \| modelo \| parser`) | ⚠️ Opcional |
| Revisar deprecaciones y actualizar imports | ⚠️ `Chroma` se deja así por decisión del grupo |
| Tokenización y lematización con NLTK, TF-IDF con Naive Bayes | — Concepto de clase, no requisito del agente |
| Agente de juguete que decide la herramienta y muestra su traza | ❌ Se cubre con la traza observable |

**Resumen:** 11 cumplidos · 6 parciales · 6 faltantes.

---

El análisis original de Nicolás, con el detalle de cada faltante y su plan en
fases, está en el historial: `git show 0d3f00b:PENDIENTES.md`. La versión
anterior de su informe de faltantes: `git show 8691493:PENDIENTES.md`.
