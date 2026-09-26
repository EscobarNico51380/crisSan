# Guía para las slides y el video (descartable)

Borrador de trabajo del grupo, **no va en la entrega**. Sirve para armar las
slides y el guion sin volver a leer todo el proyecto. Cuando esté el ReAct, se
actualizan solo las partes marcadas con **🔄 ReAct** y se generan las slides.

**Qué hay que entregar:** un video de 10 a 15 minutos (vence el **30/09**) y la
defensa oral (30/09 o 14/10), donde además preguntan sobre el TP1 (ver
`repaso_TP1.md`).

---

## 1. Qué evalúan, y dónde lo cubrimos

El enunciado pide que la presentación defienda cinco cosas, y evalúa seis
criterios. Cada bloque del video apunta a uno o más:

| Lo que piden | Bloque del video |
|---|---|
| El caso de negocio elegido | 1 · El problema |
| Cómo fue resuelto con IA · *Aplicación del marco teórico* | 2 · Arquitectura, 3 · RAG |
| *Funcionamiento del sistema* · Qué resultados obtuvieron | 4 · Demo, 6 · Resultados |
| Qué dificultades encontraron · *Identificación de dificultades* | 5 · Tres errores |
| Qué decisiones se tomaron y por qué · *Documentación técnica* | 2, 5 y 6 |
| *Creatividad y originalidad* | 1 (el caso), fotos, controles con código |

---

## 2. Estructura y tiempos (~13 minutos)

| # | Bloque | Tiempo | Slides | Quién |
|---|---|---|---|---|
| 0 | Apertura: equipo y qué es San Cris | 0:30 | 1 | *(a definir)* |
| 1 | El problema: la letra chica | 1:00 | 2 | |
| 2 | Arquitectura: el grafo y quién decide qué | 2:00 | 3-4 | |
| 3 | RAG: cómo busca en el manual | 1:30 | 5 | |
| 4 | Demo en vivo | 4:00 | (pantalla) | |
| 5 | Dificultades: tres errores, uno por pieza | 2:00 | 6-8 | |
| 6 | Resultados, costo e instalación | 1:15 | 9 | |
| 7 | Limitaciones y cierre | 0:45 | 10 | |

Si hay que recortar, se achica la demo (sacar los pasos 2-3) y el bloque de RAG.

---

## 3. Contenido de cada slide, con lo que se dice

El texto entre comillas es una propuesta de lo que se dice; conviene decirlo
con palabras propias. Los datos están verificados en `DOCUMENTACION.md`.

### Slide 1 — Portada
- **San Cris**: asistente de seguros de auto con RAG y agentes.
- TP2 de Inteligencia Artificial, UTN FRRO, comisión 502.
- Anmuth · Cabanellas · Casado · Escobar · Maragliano.

> "Somos… Hicimos San Cris, un asistente para los clientes de una aseguradora
> de autos, San Cristóbal."

### Slide 2 — El problema
- Los seguros están llenos de letra chica: coberturas, exclusiones,
  franquicias. Nadie lee el PDF de 27 páginas.
- El cliente quiere saber una cosa: **¿me cubre esto?**
- San Cris: el cliente cuenta lo que pasó (con una foto, si quiere) y recibe un
  **dictamen**: *Cubierto*, *Rechazado* o *Requiere peritaje*, con el
  fundamento del manual y un número de reclamo.

> "El cliente no tiene que saber qué plan tiene ni leer el manual: cuenta lo
> que le pasó y el sistema busca en el manual, cruza con su póliza y le
> responde si está cubierto y por qué."

### Slide 3 — Arquitectura: el grafo · 🔄 ReAct
- **Imagen: el grafo coloreado** (ver "Material a preparar"). Azul, código;
  verde, RAG; violeta, LLM; amarillo, LLM y código juntos.
- Construido con **LangGraph** (`StateGraph`): un grafo de estado con nodos y
  rutas explícitas, no un agente que decide todo solo.
- Memoria con *checkpointer* + `thread_id`; en charlas largas, un resumen.
- Stack: LangChain, LangGraph, Groq (`gpt-oss-120b`, respaldo `gpt-oss-20b`,
  visión `qwen3.8-27b`), Chroma, e5, SQLite, Streamlit.

> "Elegimos un grafo explícito porque un dictamen de cobertura tiene que seguir
> siempre los mismos pasos: la póliza, el manual, el dictamen, los controles.
> No queríamos que el modelo decidiera si hoy revisa la vigencia o no."

🔄 **ReAct:** acá se cuenta qué preguntas pasan por el agente ReAct (el modelo
elige la herramienta) y cuáles por el camino fijo del dictamen, y por qué.
Agregarlo al grafo.

### Slide 4 — Quién decide qué (la idea central)

| Pieza | Qué hace | Cómo se comporta |
|---|---|---|
| **RAG** | Busca en el manual. No decide nada | Siempre igual |
| **LLM** | Interpreta: entiende el mensaje, lee el manual, razona, redacta | Puede variar |
| **Código** | Verifica hechos: póliza, fechas, vigencia, pago, plazo, duplicados | Siempre igual |

> "El RAG es el bibliotecario que trae las páginas; el LLM es el perito que las
> lee y opina; el código es el área administrativa que revisa si la póliza está
> paga y vigente. Lo que se puede calcular lo decide el código; el modelo
> interpreta lo que no se puede calcular."

### Slide 5 — RAG: cómo busca en el manual
- El manual en PDF se corta **por los ítems numerados** (`10) Granizo`), y cada
  ítem lleva el título de su lista ("Exclusiones para robo…").
- **Búsqueda híbrida:** BM25 (palabras exactas: "granizo", "GNC") +
  embeddings `multilingual-e5-base` (el sentido: "me rompieron el vidrio"),
  combinadas con `EnsembleRetriever`.
- Si el auto tiene GNC, una **segunda búsqueda** dirigida a las exclusiones.

> "Probamos cortar el manual por cantidad de caracteres y el título quedaba
> separado del contenido. Y un buscador solo semántico no encontraba
> 'granizo': por eso la búsqueda híbrida."

### Bloque 4 — Demo (pantalla, con el recorrido de la barra lateral) · 🔄 ReAct

Usar los botones del recorrido. Pasos sugeridos y qué remarcar:

| Paso | Consulta | Qué decir |
|---|---|---|
| 2 → 3 | *¿Qué plan tengo para la patente AB123CD?* → *¿Y cuándo vence?* | Base de datos + memoria: la segunda no dice el auto |
| 4 | *¿Qué cubre el granizo?* | RAG: cita la cláusula del manual |
| 5 | *Choqué contra un poste con la AB123CD…* | Dictamen completo: cubierto con franquicia, número de reclamo |
| 6 | *¿En qué estado está mi reclamo?* | La ficha la arma el código |
| 7 → 8 | Robo del AC456BD → robo del AF987XX | Mismo plan, uno con GNC: el segundo se rechaza por una exclusión del manual |
| 9 | *Me robaron el auto AJ222NP* | Póliza vencida: la rechaza el código, no el modelo |
| 12 | Granizo sobre la AB123CD + foto de un choque | La foto no coincide con el relato: pide peritaje |

Mostrar también, debajo de una respuesta, **los tokens y el costo** del turno.

🔄 **ReAct:** sumar una o dos consultas donde se vea que el modelo elige la
herramienta (por ejemplo "¿qué reclamos tengo?", "necesito una grúa"), y, si
está hecha, la traza (qué herramienta eligió, con qué datos).

### Slides 6 a 8 — Dificultades: tres errores, uno por pieza

Resolvimos 34 problemas (tabla completa en `DOCUMENTACION.md`). Estos tres
muestran cómo pensamos cada uno:

**6 · Falla del RAG: el GNC.** El robo de un auto con GNC salía cubierto, aunque
el manual lo excluye. El modelo razonaba bien: **la regla nunca le llegaba**,
porque nadie escribe "tengo GNC" al denunciar un robo. Además, cada ítem del
manual perdía el título de su lista y la exclusión se aplicaba donde no
correspondía. El arreglo fue en la búsqueda, no en el modelo.

**7 · Falla del LLM: el Cruze.** El robo del Chevrolet Cruze con GNC salió
cubierto 1 vez de 5. **La regla le llegaba** (se verificó) y aun así razonó
mal. Los modelos no son completamente deterministas, ni con `temperature=0`.
Decidimos aceptarlo y explicarlo.

**8 · Tarea mal asignada: la póliza vencida.** Salía cubierta: el modelo tenía
las dos fechas a la vista y no las comparaba. **Comparar fechas nunca fue
trabajo para el modelo**: pasó al código, y desde entonces no falla. Con la
misma idea: el pago, el plazo de denuncia de 72 horas y los reclamos
duplicados.

> "Cada error nos enseñó a qué pieza le corresponde cada tarea."

Otras que se pueden nombrar si sobra tiempo: la foto de un choque con relato
de granizo salía cubierta (se resolvió con una llamada aparte: una tarea por
llamada); el modelo de visión confundía granizo con nieve (se probaron tres
redacciones de sus instrucciones); el cupo diario se agotaba (modelo de
respaldo automático).

### Slide 9 — Resultados
- **Batería final con el modelo principal:** 23 de 25 correctos; de los 2
  restantes, uno era un error de la verificación automática y el otro, el
  Cruze.
- **Costo:** un dictamen usa ~9.100 tokens, unos US$ 0,002: **~500 dictámenes
  por dólar**. Con el plan gratuito, unos 20 dictámenes o 45 consultas por día.
- **Fotos:** un modelo de visión describe el daño; no decide la cobertura.
- **Instalación con un solo comando:** `python iniciar.py`. Probado desde cero:
  10 minutos.

🔄 **ReAct:** sumar el resultado de sus pruebas.

### Slide 10 — Limitaciones y cierre
- El dictamen puede variar (el Cruze).
- El modelo de visión puede equivocarse: por eso solo describe.
- El granizo o el parabrisas de un auto con GNC: el manual es ambiguo.
- El plazo de denuncia no contempla caso fortuito o fuerza mayor.

> "La lección más grande: el modelo de lenguaje es muy bueno interpretando y
> malo verificando. Un sistema confiable combina las dos cosas: que el código
> decida lo que se puede calcular y el modelo interprete lo demás."

---

## 4. Material a preparar

- **El grafo coloreado:** copiar el bloque `mermaid` de la sección "Quién decide
  qué" de `DOCUMENTACION.md` en https://mermaid.live y exportar PNG o SVG.
  🔄 ReAct: actualizarlo antes.
- **Capturas** (por si la demo en vivo falla): respuesta del GNC rechazado, la
  póliza vencida, la foto que no coincide, la ficha del reclamo, el costo
  debajo de una respuesta.
- **Tabla de las tres piezas** (slide 4) y **stack** (slide 3).

## 5. Antes de grabar

- [ ] Grabar con el **cupo renovado** del modelo principal (cada demo completa
      usa ~80.000 tokens) y el selector de la web en `gpt-oss-120b`.
- [ ] **Vaciar la tabla de reclamos** (las pruebas quedan como precedentes y
      como duplicados):
      `python -c "import sqlite3;c=sqlite3.connect('data/seguros_tp.db');c.execute('DELETE FROM siniestros');c.commit()"`
- [ ] Reiniciar la conversación en la web.
- [ ] Zoom del navegador al 125 % para que se lea en el video.
- [ ] **Que no se vea la API key** (ni el `.env` ni la terminal donde se pegó).
- [ ] Ensayar una vez la demo: los dictámenes tardan unos segundos.
- [ ] Grabación: la barra de juegos de Windows (`Win + Alt + R`) u OBS.

## 6. Para generar las slides con otro asistente

Pasarle este archivo y la imagen del grafo, con un pedido como:

> "Armá una presentación de 10 slides con esta guía (una slide por cada
> «Slide N» de la sección 3), estilo sobrio, poco texto por slide, con la
> imagen del grafo en la slide 3. Las citas entre comillas son lo que se dice:
> ponelas en las notas del orador, no en la slide."

## 7. Qué actualizar cuando esté el ReAct

- [ ] Slide 3: qué pasa por el ReAct y qué por el camino fijo, y por qué.
- [ ] El grafo coloreado (en `DOCUMENTACION.md` y la imagen).
- [ ] La demo: consultas donde el modelo elige la herramienta, y la traza.
- [ ] Slide 9: resultados de las pruebas del ReAct.
- [ ] Revisar los tiempos: si la demo crece, recortar el bloque de RAG.
