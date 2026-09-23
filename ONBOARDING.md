# Cómo ejecutar San Cris y hablar con el agente

Guía corta para levantar el asistente de seguros (RAG + LangGraph + Groq) y hacerle preguntas útiles. El detalle de arquitectura está en `DOCUMENTACION.md`.

## Quick path

1. Entrá al repo y creá el entorno:

```bash
cd /home/nicolas/proyectos/tpIA
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. Copiá `.env.example` a `.env` y pegá tu key de [Groq](https://console.groq.com/keys):

```bash
cp .env.example .env
# Editá .env: GROQ_API_KEY=gsk_...
```

3. Arrancá la UI (recomendado) o la consola:

```bash
.venv/bin/streamlit run app.py
# o
.venv/bin/python main.py
```

La primera vez crea `data/seguros_tp.db` y vectoriza el PDF en Chroma (`data/chroma_db/`). Puede tardar un rato por la descarga del modelo de embeddings.

## Datos de prueba (usá estos)

El agente solo “conoce” clientes y pólizas sintéticos. Sin patente o DNI no puede emitir dictamen ni consultar una póliza concreta.

| Qué | Valor | Qué hay detrás |
|-----|--------|----------------|
| Titular | Carlos Perez | DNI `39123456` |
| Auto 1 | patente `AB123CD` | Ford Focus 2024, plan **D c/fran** (todo riesgo), franquicia fija $460.000, con rastreador |
| Auto 2 | patente `AF987XX` | Peugeot 208 2012, plan **C+**, con GNC, sin rastreador |
| Pólizas | `POL-1001` / `POL-1002` | Ambas al día, vigentes en 2026–2027 |

## Cómo preguntarle

Escribí en lenguaje natural. Incluí **patente** (`AB123CD` / `AF987XX`) o **DNI** (`39123456`) cuando la consulta sea de un asegurado. La memoria del chat recuerda patente/DNI entre turnos.

| Qué querés probar | Ejemplo |
|-------------------|---------|
| Charla | `Hola, ¿quién sos?` |
| Plan / franquicia / pago | `¿Qué plan tengo para la patente AB123CD?` |
| Listar pólizas por DNI | `Mi DNI es 39123456, ¿qué pólizas tengo?` |
| Cláusulas del manual (RAG, sin póliza) | `¿Qué es la franquicia?` / `¿Qué cubre el granizo?` |
| Siniestro (dictamen + reclamo) | `Choqué contra un poste con la AB123CD` |
| Siniestro en C+ (a menudo distinto al D) | `Choqué contra un poste con la AF987XX, ¿me cubre?` |
| Reclamo (después de registrar uno) | `¿En qué estado está mi reclamo #1?` |

### Mini script de demo (2–3 minutos)

1. `Hola, ¿quién sos?`
2. `¿Qué plan tengo para la patente AB123CD?`
3. `Me rompieron el vidrio lateral, ¿me cubre?` (usa la patente recordada)
4. Anotá el número de reclamo y preguntá: `¿Estado del reclamo N?`

En Streamlit, **Reiniciar conversación** borra memoria y `thread_id`. En consola, `salir` / Ctrl+C termina; un hilo nuevo implica volver a correr `python main.py`.

## Qué esperar (y qué no)

- **Dictamen** cubierto / rechazado / peritaje, con cláusulas del PDF y, si aplica, número de reclamo en SQLite.
- Preguntas de dominio (`¿qué es CLEAS?`) van al **manual**, no a la póliza del cliente.
- Patentes/DNI inventados: el agente no inventa pólizas; debería decir que no hay datos.
- Fuera de seguros (dólar, fútbol): no es el caso de uso.

## Si algo falla

| Síntoma | Qué hacer |
|---------|-----------|
| Error de API / “no se encontró GROQ_API_KEY” | Revisá `.env` en la raíz del repo y reiniciá Streamlit |
| Primera corrida lenta | Normal: embeddings HuggingFace + indexado del PDF |
| Respuesta vacía o 429 | Cupo/rate limit de Groq; esperá y reintentá |
| Base vieja / datos raros | Borrá `data/seguros_tp.db` y volvé a correr (se recrea) |

## Checklist

- [ ] `.venv` instalado con `requirements.txt`
- [ ] `.env` con `GROQ_API_KEY` válida
- [ ] Streamlit o `main.py` responden un “Hola”
- [ ] Consulta con `AB123CD` o `AF987XX` devuelve plan/titular
- [ ] Un siniestro genera dictamen y (si corresponde) número de reclamo

## Next step

Para defender el TP: arquitectura, tools y decisiones están en `DOCUMENTACION.md`. Para una batería automática de casos: `.venv/bin/python test_casos.py` (consume cuota de Groq).
