# Pendientes — San Cris (TP2)

Una sola lista del grupo. Cuando algo se termina, se marca ✅ y se anota el
commit. Al final está el estado del sistema frente a lo que piden los notebooks
de la cátedra.

**Fechas:** no hay video. La presentación es **oral, con PowerPoint y
demostración en vivo**, en la defensa: el 30/09 o el 14/10 (turno tarde/noche),
o el 01/10 o el 08/10 (turno mañana); incluye 1-2 preguntas sobre el TP1.

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
| Al consultar un reclamo respondía solo "está en estado Iniciado" | ✅ La ficha completa la arma el código, sin el LLM | `469418c`, `35fbdc5` |
| Una póliza vencida sale cubierta | ✅ La vigencia y el pago se controlan con código, sin el LLM | `a48070e` |
| "Me robaron el auto AJ222NP el 10 de mayo de 2026" se clasificó como consulta y no como denuncia | ✅ Con el modelo principal se clasifica bien (era el modelo chico) | batería final |
| El plazo de denuncia se evalúa distinto según cómo se redacte | ✅ Lo decide el código mirando la base: si ya estaba denunciado muestra el reclamo; si no y pasaron más de 72 horas, rechaza. También evita reclamos duplicados | `32decb9` |
| El robo del auto todo riesgo con GNC (AD789EF) salió *Cubierto* 1 de 5 veces | ✅ Decidido: se acepta y se explica en la defensa | Es un límite real de los LLM: aun con `temperature=0`, no son completamente deterministas |
| Cuando no hay precedentes del mismo producto, se muestran los de otros productos | ✅ Decidido: se deja como está | Se probó cargando el precedente sospechoso y el dictamen salió bien: no hay evidencia de que confunda |
| La consola de Streamlit muestra decenas de errores *"No module named 'torchvision'"* | ⏳ Diagnosticado, sin acción por ahora | No rompe nada: el vigilante de archivos de Streamlit inspecciona todos los módulos cargados, y ~100 módulos de imágenes de `transformers` (que no usamos) fallan al cargarse sin `torchvision`. Streamlit atrapa cada error y sigue. Reproducido fuera de Streamlit: 102 módulos fallan. Probablemente es lo que vio Nicolás y por lo que instaló `torchvision` (no una diferencia de Linux): preguntarle. Arreglo propuesto: `fileWatcherType = "none"` en `.streamlit/config.toml`; si se confirma, sacar `torchvision` también de Linux |
| Si el último paso recibe una respuesta vacía del modelo, el cliente ve "¿Podrías reformular tu consulta?" aunque su reclamo ya quedó registrado | ✅ Un reintento; y en una denuncia la ficha la arma siempre el código (en la prueba del 27/09 el modelo devolvió vacío dos veces y quedó la ficha) | `5a41204` |
| La traza de la póliza vencida decía "Cubierto" y el chat "Rechazado" | ✅ Muestra el dictamen final y, si un control lo cambió, lo que había dicho el modelo | `5a41204` |
| Un pedido de grúa se derivaba a un asesor como "no reconocido" | ✅ Tipo "Asistencia mecánica" (dictamen Asistencia 24) | `5a41204` |
| Al registrar una denuncia, el modelo dio un número de reclamo equivocado, y en la vencida enterró el rechazo | ✅ La ficha la arma el código; el modelo agrega los pasos a seguir | `5a41204` |
| "¿Cómo avanzo para terminar mi reclamo?" mostraba la ficha del reclamo una y otra vez, en vez de decir que no puede gestionarlo | ✅ Vitto: los trámites van al ReAct, que lo dice con honestidad y deriva al 0810-222-8887 (probado el 27/09 con eso y con "¿me pueden dar de baja la póliza?") | `b16be68` |
| Una consulta por el ReAct superó el límite de tokens por minuto (413) | ✅ 6 fragmentos por búsqueda y una sola búsqueda por pregunta | `5a41204` |
| La terminal de la instalación muestra avisos que parecen errores (versión nueva de pip, Hugging Face sin token, *symlinks* en Windows) | ⏳ Propuesta: apagarlos con variables de entorno en `iniciar.py` | Vistos en la prueba desde cero del 25/09 |
| La clase `Chroma` que usamos está deprecada | Se deja así: el proyecto dura un mes | |

## Features

En orden sugerido (el plan de Nicolás): primero poder cambiar de modelo, porque
el agente ReAct y las imágenes se apoyan en eso.

| Feature | Estado | Notas |
|---|---|---|
| Costo estimado de cada consulta, visible en el chat | ✅ | `4906c02` |
| Recorrido sugerido en la web: botones con las consultas recomendadas | ✅ | `2d828c8` |
| Más autos de prueba en la base | ✅ 6 autos nuevos, cada uno prueba una sola cosa | `a7f8f3f` |
| Correr con el modelo grande los 5 casos de autos nuevos que se cortaron por el cupo | ✅ Batería final: 23 de 25, todo con el modelo principal | `1d9900b` |
| Modelo de respaldo automático cuando el principal agota su cupo, sin reintentar el agotado | ✅ | `c72e8af` |
| Elegir el modelo desde la web (el otro queda de respaldo) | ✅ | `7e986c3` |
| Poder cambiar de modelo desde la configuración (`make_llm`, como en el notebook de agentes) | 🔶 Casi | `llm.py` ya crea los modelos con `make_llm` y los nombres están en `config.py`. Falta poder elegir otro proveedor (OpenAI, Ollama) |
| Que el LLM elija las herramientas (patrón ReAct con `create_agent`), dejando fijo el camino del dictamen | ✅ Nicolás: `d637ff4`, `bea309e`; merge `40abe81` | Todas las consultas (póliza, DNI, reclamos, cuánto paga, manual) pasan por el ReAct; las denuncias, por el camino fijo. La consulta por patente se pasó al ReAct (`5a41204`) y las exclusiones por GNC, a la herramienta de búsqueda, así no se pierden. Ver `DOCUMENTACION.md` |
| Herramienta nueva: listar los reclamos de un cliente ("¿qué reclamos tengo?") | ✅ `listar_reclamos`, por patente o DNI | `5a41204` |
| Herramienta nueva: calcular cuánto paga el cliente (costo de la reparación menos la franquicia) | ✅ `calcular_pago_cliente`: la cuenta la hace el código | `5a41204` |
| Herramienta nueva: pedir una grúa o asistencia | ⏳ Queda como limitación | Sería una herramienta que escribe, y el ReAct solo tiene de lectura. Hoy un pedido de grúa se registra como reclamo "Derivado a Asistencia 24" |
| Probar el ReAct con el modelo principal: la batería completa (`test_casos.py`) y el recorrido de la web | ⏳ | El 27/09 la segunda vuelta de pruebas la respondió casi toda el modelo de respaldo (cupo agotado) |
| Mostrar en la web de dónde sale cada respuesta: herramienta elegida, con qué datos, qué devolvió, y los fragmentos del manual con su puntaje | ✅ | Pestaña "🔍 Cómo llegué a esto" (Razonamiento / Herramientas / Fuentes) en el chat. `rag.buscar_clausulas_con_evidencia` devuelve los chunks con sección, página y similitud coseno |
| Fijar el tipo de siniestro a los del manual, y derivar a un asesor si no coincide | ✅ | `DictamenSiniestro.tipo_siniestro` es un `Literal` con los 11 tipos del manual, `Asistencia mecánica` (sumado en `5a41204`) y `No reconocido`; en ese caso `nodo_derivar_humano` deriva al 0810-222-8887 y no registra |
| Interpretar imágenes enviadas por el cliente | ✅ Una foto por mensaje en la web; el modelo de visión describe y el dictamen la usa como evidencia | `9f7e9c3` |
| Probar los dictámenes con foto con el modelo principal | 🔶 | 6 casos bien con el principal el 25/09 (ver `DOCUMENTACION.md`). Faltan el choque con relato de granizo (paso 12 del recorrido) y el parabrisas del auto con GNC, que los respondió el respaldo |
| Probar con fotos propias del grupo | ⏳ | Las de ejemplo son de Wikimedia Commons |
| Exclusiones por gas licuado (GLP) y por uso para transporte de pasajeros | ⏳ A confirmar | El manual las tiene; se sumarían con el mismo mecanismo que el GNC |
| Historial que sobreviva al cerrar la app (`SqliteSaver`) | ⏳ Baja prioridad | Suma una dependencia |
| Pruebas de las herramientas sin usar el LLM | ⏳ Baja prioridad | Como las del notebook de agentes; hoy `test_casos.py` pasa siempre por el LLM |
| Límite de recursión en cada consulta (`recursion_limit`) | ⏳ Baja prioridad | El ReAct usa el de LangGraph por defecto, que en la versión instalada es 10.007 pasos: en la práctica, sin tope. En las pruebas nunca pasó de 2 herramientas. El notebook de agentes de la cátedra usa 12 |
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
  En la prueba de fotos del 25/09 el modelo de respaldo dio *Rechazado* y
  *Cubierto* para el mismo parabrisas del AF987XX: otra razón para fijarlo.

## Entregables

| Entregable | Estado |
|---|---|
| Slides de la presentación (PowerPoint) | ✅ La de Gemini (`Presentacion alternativa - hecha con Gemini.pdf`, 10 diapositivas), con el guion por integrante y la lista de cambios a hacerle en `guion_defensa_San_Cris_Gemini.md` (fuera del repo) |
| Mostrar el grafo coloreado según quién decide cada paso (código, RAG o LLM), con el recorrido de una consulta y los tres errores de ejemplo | ✅ `grafo.png`: horizontal, fondo blanco, con la leyenda de colores (fuente en `grafo.mmd`). `grafo2.png` es otra versión, con fondo oscuro y sin leyenda |
| Demo en vivo | 🔶 Pruebas, qué señalar en la pantalla y qué decir en el guion; reparto de quién habla hecho. Falta ensayarla |
| Vaciar la tabla de siniestros antes de presentar (las pruebas quedan como precedentes) | ⏳ |
| Repaso del TP1 | 🔶 `repaso_TP1.md`: preguntas probables con respuesta, los conceptos de la cátedra y dos inconsistencias del notebook para tener preparadas. Falta que lo lea el grupo y que cada uno sume lo suyo si su TP1 fue distinto |

## Cierre (lo último antes de entregar)

| Tarea | Estado |
|---|---|
| `README.md` con los integrantes, y `iniciar.py`: instala y abre el chat con un solo comando | ✅ | `d291f74` |
| Reorganizar la documentación: se borra `ONBOARDING.md` (lo cubre el README) y `SETUP.md` pasa a ser `AGENTS.md`, la guía interna para el equipo y para asistentes de IA (`CLAUDE.md` la importa) | ✅ | `3085cb5` |
| Limpiar la documentación que se entrega: sin hashes de commits ni ramas, los pendientes internos pasados a limitaciones ("Lo que no se probó"), la póliza vencida del modelo chico contada en pasado, el paso 9 del recorrido corregido (el modelo dictamina y el control lo corrige) y el comentario de `.streamlit/config.toml` sin mencionar un video | ✅ |
| Armar el zip de la entrega. **Va:** `README.md`, `DOCUMENTACION.md`, los `.py` (incluidas las dos pruebas), `requirements*.txt`, el PDF del manual, `ejemplos/`, `.streamlit/`, `.env.example`, `grafo.png` y el `.env` con una key de Groq de una cuenta nueva dedicada a la entrega (1-2 meses de vida, revocarla después). **No va:** `.git`, `.venv`, `__pycache__`, `data/` completa (`iniciar.py` arma la base limpia y el índice; así lo probó la instalación desde cero), los documentos internos (`AGENTS.md`, `CLAUDE.md`, `PENDIENTES.md`, `repaso_TP1.md`, `guia_presentacion.md`, `IdeaTP.md`), el enunciado, `notebooks_clases/`, `grafo.mmd`, `grafo2.png` y el PDF de la presentación. Comprimir con `tar -a` y no con `Compress-Archive` (el de PowerShell 5.1 guarda rutas con `\` y se rompen en Linux). El README ya aclara que, si viene el `.env`, no hay que hacer nada | ⏳ |
| Prueba previa del grupo: clonar el repo en una carpeta nueva e instalar con `python iniciar.py`, en Windows y en Linux (Nicolás), y reportar sistema, versión de Python, tiempo y errores | ⏳ Windows ✅ el 25/09 (10 minutos, sin errores; ver `DOCUMENTACION.md`). Falta Linux |
| Probar la instalación desde cero con el zip, siguiendo solo el README, en Windows y en Linux | ⏳ |

---

## Estado frente a los notebooks de la cátedra

Análisis de Nicolás, con los estados actualizados. Los notebooks, pasados a
`.py`, están en `notebooks_clases/`.

**Leyenda:** ✅ cumple · ⚠️ parcial · ❌ falta · — no aplica

### Agentes (`copia_de_nlp_4_langchain_agentes`)

| Requisito | Estado |
|---|---|
| `create_agent`: el modelo decide si responder o llamar una herramienta (ReAct) | ✅ |
| Herramientas con `@tool` (descripción desde el docstring, esquema desde los tipos) | ✅ |
| Herramientas verificadas localmente, sin LLM | ⚠️ Se prueban a través del agente |
| Backend separado, modelo intercambiable (`make_llm`) | ⚠️ `llm.make_llm` crea todos los modelos; falta soportar otro proveedor (OpenAI/Ollama) |
| Prompt de sistema con rol y reglas | ✅ |
| Memoria con checkpointer + `thread_id` | ✅ |
| Traza observable: herramienta, argumentos, resultado | ✅ |
| `recursion_limit` en la consulta | ⚠️ |
| Modelo con soporte de *tool calling* | ✅ Lo usa el subagente ReAct |

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
| Mostrar las fuentes y los puntajes de la búsqueda | ✅ Pestaña Fuentes + similitud coseno por chunk |
| Reintentos con espera creciente ante errores temporales | ✅ |

### Conceptos (`nlp_1_conceptos`)

| Requisito | Estado |
|---|---|
| LCEL (`prompt \| modelo \| parser`) | ⚠️ Opcional |
| Revisar deprecaciones y actualizar imports | ⚠️ `Chroma` se deja así por decisión del grupo |
| Tokenización y lematización con NLTK, TF-IDF con Naive Bayes | — Concepto de clase, no requisito del agente |
| Agente de juguete que decide la herramienta y muestra su traza | ✅ Se cubre con el subagente ReAct y la traza observable |

**Resumen:** 16 cumplidos · 6 parciales · 1 faltante.

---

El análisis original de Nicolás, con el detalle de cada faltante y su plan en
fases, está en el historial: `git show 0d3f00b:PENDIENTES.md`. La versión
anterior de su informe de faltantes: `git show 8691493:PENDIENTES.md`.
