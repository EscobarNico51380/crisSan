# San Cris — Asistente de seguros con RAG y agentes

**Trabajo Práctico N°2 — Inteligencia Artificial**
UTN Facultad Regional Rosario · Comisión 502

**Integrantes:** Anmuth · Cabanellas · Casado · Escobar · Maragliano

---

## Qué es

San Cris es un asistente conversacional para los clientes de una aseguradora de
autos (San Cristóbal). El cliente escribe en lenguaje natural y el sistema:

- responde dudas sobre el manual de la póliza, buscando las cláusulas
  relevantes (**RAG** sobre el PDF del manual);
- consulta los datos de la póliza del cliente en una base de datos, lista sus
  reclamos y calcula cuánto paga de una reparación;
- analiza un siniestro y **dictamina si está cubierto**, citando las cláusulas
  que lo fundamentan, y registra el reclamo;
- mira la **foto** del daño, si el cliente la adjunta, y la usa como evidencia;
- recuerda la conversación, así que entiende preguntas de seguimiento;
- muestra, debajo de cada respuesta, **cómo llegó a ella**: qué herramientas
  usó, con qué datos, y qué fragmentos del manual consultó.

Las consultas las resuelve un **agente ReAct**, que elige por sí mismo qué
herramienta usar; las denuncias siguen siempre los mismos pasos, con controles
hechos por el código.

Está construido con **LangChain** y **LangGraph**, y usa el modelo
`gpt-oss-120b` a través de **Groq** (y `qwen3.8-27b` para mirar las fotos).

---

## Cómo probarlo: un solo comando

**Requisito:** Python 3.11 a 3.13 (probado con 3.13). Se descarga de
[python.org](https://www.python.org/downloads/). En Windows, durante la
instalación, tildar **"Add python.exe to PATH"**.

Abrir una terminal en esta carpeta y ejecutar:

```
python iniciar.py
```

En Windows también funciona `py iniciar.py`.

**La primera vez**, el script:

1. muestra qué va a instalar y pide confirmación;
2. instala todo (descarga unos 1,5 GB y ocupa unos 2,6 GB; tarda entre 5 y
   20 minutos, según la conexión: en nuestra prueba desde cero, 10);
3. verifica la API key;
4. abre el chat en el navegador.

**Las siguientes veces** abre el chat directamente, en segundos.

Para cerrar el chat: `Ctrl+C` en la terminal.

---

## Qué se instala en la computadora

Todo queda **dentro de esta carpeta**:

| Carpeta | Qué contiene | Tamaño |
|---|---|---|
| `.venv/` | Entorno de Python aislado con las librerías del proyecto (LangChain, LangGraph, PyTorch, Streamlit…) | ~1,5 GB |
| `data/modelos/` | Modelo de embeddings `multilingual-e5-base`, para buscar en el manual | ~1,1 GB |
| `data/` | Base de datos de prueba (SQLite) e índice del manual (Chroma) | ~3 MB |

- Usa el Python que ya está instalado; no instala otro.
- No necesita permisos de administrador ni cambia la configuración del sistema.
- No deja copias en la caché de pip ni en la de modelos del usuario.
- La única excepción es Streamlit, que crea una carpeta `.streamlit` en el
  usuario con dos archivos de menos de 100 bytes.

**Para desinstalar:** borrar esta carpeta (y, si se quiere, la carpeta
`.streamlit` del usuario).

---

## La API key

El sistema usa el plan gratuito de Groq, que necesita una API key.

- **Si la carpeta incluye un archivo `.env` con la key**, no hay que hacer
  nada: el script la verifica y sigue.
- **Si no**, el script la pide. Se consigue gratis y sin tarjeta en
  [console.groq.com/keys](https://console.groq.com/keys) y empieza con `gsk_`.
  El script la guarda en `.env`.

**Cupo diario:** el plan gratuito alcanza para unas 20 denuncias de siniestro o
45 consultas por día con el modelo principal. Si se agota, el sistema pasa solo
a un modelo de respaldo (`gpt-oss-20b`), que tiene su propio cupo. Si se
agotan los dos, el chat avisa cuánto hay que esperar.

Debajo de cada respuesta se muestra cuántos tokens usó y cuánto costaría en el
plan pago de Groq (con el gratuito, el costo real es cero).

---

## Recorrido sugerido

La barra lateral del chat tiene un **recorrido sugerido**: doce consultas en
orden, cada una con un botón que la envía. Debajo de cada botón se explica qué
muestra. También se puede escribir cualquier consulta en el chat.

| Paso | Consulta | Qué muestra |
|---|---|---|
| 1 | Hola, ¿quién sos? | Se presenta y cuenta qué puede hacer |
| 2 | ¿Qué plan tengo para la patente AB123CD? | Consulta la póliza en la base de datos |
| 3 | ¿Y cuándo vence? | Memoria: sigue hablando del mismo auto |
| 4 | ¿Qué cubre el granizo? | Busca la respuesta en el manual (RAG) |
| 5 | Choqué contra un poste con la AB123CD, se abolló la puerta | Dictamen: cubierto con franquicia, y registra el reclamo |
| 6 | ¿En qué estado está mi reclamo? | Consulta el reclamo recién registrado |
| 7 | Me robaron el auto AC456BD | Robo con plan C+ y sin GNC: cubierto |
| 8 | Me robaron el auto AF987XX | El mismo robo y el mismo plan, pero con GNC: rechazado por una exclusión del manual |
| 9 | Me robaron el auto AJ222NP | Póliza vencida: aunque el modelo de lenguaje diga "Cubierto", la rechaza un control del código |
| 10 | Mi DNI es 30555777 | Busca por DNI y lista las pólizas de la titular |
| 11 | Me cayó granizo sobre la AC456BD 📷 | Con foto de granizo: el modelo de visión describe el daño y coincide con el relato |
| 12 | Me cayó granizo sobre la AB123CD 📷 | Con la foto de un choque frontal: no coincide con el relato, así que pide peritaje |

Los pasos con 📷 envían, además, una foto de ejemplo. El recorrido completo usa
unos 80.000 tokens, menos de la mitad del cupo diario.

---

## Fotos

Se puede **adjuntar una foto del daño** con el clip del chat (una por mensaje;
JPG, PNG o WEBP). Sirve con texto (*"Me cayó granizo sobre la AC456BD"*) o sin
él (*"¿Me cubre esto?"*, o solo la foto).

- Un modelo de visión (`qwen3.8-27b`) **describe** la foto: qué parte del auto,
  qué tipo de daño y qué tan grave. **No decide la cobertura.**
- El dictamen lo sigue emitiendo el modelo principal, que usa esa descripción
  como evidencia. Si la foto no coincide con el relato, pide **peritaje**.
- La foto **no se guarda**: con el reclamo queda solo su descripción.
- Si la foto no se puede analizar, el chat lo avisa y responde con el relato.

En `ejemplos/fotos/` hay tres fotos para probar (granizo, parabrisas roto y
choque frontal), con sus créditos. La descripción es automática y puede tener
errores: ver `DOCUMENTACION.md`.

---

## Datos de prueba

El sistema solo conoce estos clientes y pólizas, que son ficticios. Cada auto
está pensado para probar una situación distinta.

| Patente | Auto | Titular (DNI) | Póliza | Qué prueba |
|---|---|---|---|---|
| AB123CD | Ford Focus 2024 | Carlos Perez (39123456) | Todo riesgo, franquicia fija de $460.000 | Cubre casi todo |
| AF987XX | Peugeot 208 2012, con GNC | Carlos Perez (39123456) | C+ (terceros completos) | Exclusiones por GNC: robo e incendio; granizo y cristales, peritaje |
| AC456BD | VW Gol Trend 2018 | Laura Gomez (30555777) | C+, sin GNC | El mismo plan que el anterior, sin GNC |
| AD789EF | Chevrolet Cruze 2022, con GNC | Laura Gomez (30555777) | Todo riesgo | La exclusión por GNC se aplica aun con el mejor plan |
| AE321GH | Fiat Uno 2008 | Laura Gomez (30555777) | Plan A (solo daños a terceros) | No cubre daños al propio auto |
| AG654JK | Renault Kangoo 2015 | Martin Diaz (28444666) | Plan B (daños totales) | Cubre robo total, no daños parciales |
| AH111LM | Toyota Etios 2019 | Martin Diaz (28444666) | C+, **impaga** | Sin pago al día no hay cobertura |
| AJ222NP | Honda Fit 2014 | Martin Diaz (28444666) | C+, **vencida** el 01/06/2026 | Póliza vencida |

---

## Qué esperar (y qué no)

- **Solo conoce los clientes y las pólizas de la tabla anterior**, que son
  ficticios. Con una patente o un DNI inventados, responde que no los encuentra;
  no inventa pólizas.
- **Responde sobre seguros de auto de San Cristóbal.** Si se le pregunta otra
  cosa (el dólar, fútbol), lo aclara y ofrece lo que sí puede hacer.
- **Para dictaminar un siniestro necesita saber de qué auto se habla.** Si la
  patente no está en el mensaje ni en la conversación, la pide.
- **Cada denuncia queda registrada** como reclamo, con número, y se puede
  consultar después (*"¿En qué estado está mi reclamo?"*, *"¿Qué reclamos
  tengo?"*).
- **Puede calcular cuánto paga el cliente** de una reparación cubierta, según
  la franquicia de su póliza (*"Si arreglar la puerta sale 800.000 pesos,
  ¿cuánto pago yo?"*).
- **"🔍 Cómo llegué a esto"**, debajo de cada respuesta, muestra las
  herramientas que usó, qué devolvieron y los fragmentos del manual.
- **El botón "Reiniciar conversación"** de la barra lateral borra la memoria de
  la charla (y el contador de costo) y empieza una nueva. Los reclamos
  registrados se conservan.

---

## Otras formas de usarlo

| Comando | Qué hace |
|---|---|
| `python iniciar.py --consola` | Abre el chat en la terminal, en vez del navegador |
| `python iniciar.py --solo-instalar` | Instala y termina, sin abrir el chat |
| `python iniciar.py --si` | Instala sin pedir confirmación |

---

## Documentación

- [`DOCUMENTACION.md`](DOCUMENTACION.md): arquitectura, decisiones de diseño,
  las dificultades que encontramos y cómo las resolvimos, y las pruebas con sus
  resultados.

---

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `python` no se reconoce como comando, o se abre la Microsoft Store | Python no está instalado o no está en el PATH. Instalarlo desde python.org tildando "Add python.exe to PATH", o probar con `py iniciar.py` |
| Durante la instalación aparecen avisos: *"A new release of pip is available"*, *"You are sending unauthenticated requests to the HF Hub"* o *"cache-system uses symlinks"* | No son errores: se pueden ignorar. La instalación sigue sola |
| Se cortó la descarga o la instalación | Volver a ejecutar `python iniciar.py`: retoma desde donde quedó |
| El chat dice *"El sistema no puede responder consultas en este momento. Probá de nuevo en N minutos."* | Se agotó el cupo diario de los dos modelos. Esperar lo que indica |
| *"El chat no pudo arrancar"* | Probablemente ya hay otra ventana del chat abierta. Cerrarla (`Ctrl+C` en su terminal) y volver a ejecutar |
| La primera consulta tarda | Es normal: carga el modelo de embeddings. Las siguientes son más rápidas |
