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
- consulta los datos de la póliza del cliente en una base de datos;
- analiza un siniestro y **dictamina si está cubierto**, citando las cláusulas
  que lo fundamentan, y registra el reclamo;
- recuerda la conversación, así que entiende preguntas de seguimiento.

Está construido con **LangChain** y **LangGraph**, y usa el modelo
`gpt-oss-120b` a través de **Groq**.

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
2. instala todo (descarga unos 3 GB; tarda entre 5 y 20 minutos, según la
   conexión);
3. verifica la API key;
4. abre el chat en el navegador.

**Las siguientes veces** abre el chat directamente, en segundos.

Para cerrar el chat: `Ctrl+C` en la terminal.

---

## Qué se instala en la computadora

Todo queda **dentro de esta carpeta**:

| Carpeta | Qué contiene | Tamaño |
|---|---|---|
| `.venv/` | Entorno de Python aislado con las librerías del proyecto (LangChain, LangGraph, PyTorch, Streamlit…) | ~1,7 GB |
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

La barra lateral del chat tiene un **recorrido sugerido**: diez consultas en
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
| 9 | Me robaron el auto AJ222NP | Póliza vencida: la rechaza un control del sistema, sin el modelo de lenguaje |
| 10 | Mi DNI es 30555777 | Busca por DNI y lista las pólizas de la titular |

El recorrido completo usa unos 60.000 tokens, menos de un tercio del cupo
diario.

---

## Datos de prueba

El sistema solo conoce estos clientes y pólizas, que son ficticios. Cada auto
está pensado para probar una situación distinta.

| Patente | Auto | Titular (DNI) | Póliza | Qué prueba |
|---|---|---|---|---|
| AB123CD | Ford Focus 2024 | Carlos Perez (39123456) | Todo riesgo, franquicia fija de $460.000 | Cubre casi todo |
| AF987XX | Peugeot 208 2012, con GNC | Carlos Perez (39123456) | C+ (terceros completos) | Exclusiones por GNC: robo e incendio |
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
  consultar después.
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
| Se cortó la descarga o la instalación | Volver a ejecutar `python iniciar.py`: retoma desde donde quedó |
| El chat dice *"El sistema no puede responder consultas en este momento. Probá de nuevo en N minutos."* | Se agotó el cupo diario de los dos modelos. Esperar lo que indica |
| *"El chat no pudo arrancar"* | Probablemente ya hay otra ventana del chat abierta. Cerrarla (`Ctrl+C` en su terminal) y volver a ejecutar |
| La primera consulta tarda | Es normal: carga el modelo de embeddings. Las siguientes son más rápidas |
