# Pendientes — San Cris (TP2)

Una sola lista del grupo. Cuando algo se termina, se marca ✅ y se anota el
commit.

**Fechas:** video de 10-15 minutos el **30/09** (todos los grupos). Defensa
oral el 30/09 o el 14/10 (turno tarde/noche), o el 01/10 o el 08/10 (turno
mañana); incluye 1-2 preguntas sobre el TP1.

**Formato de entrega (confirmado por el profesor):** proyecto de varios `.py`
con la documentación, siempre que incluya las instrucciones para reproducirlo.

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
| Al saludar, responde pidiendo la patente | ⏳ Antes del video | |
| La clase `Chroma` que usamos está deprecada | Se deja así: el proyecto dura un mes | |

## Features

| Feature | Estado | Notas |
|---|---|---|
| Costo estimado de cada consulta, visible en el chat | ✅ | `4906c02` |
| Que el LLM elija las herramientas (patrón ReAct), dejando fijo el camino del dictamen | ⏳ | Lo que más suma en "aplicación del marco teórico". Falta decidir con qué armarlo (`create_agent`, que es el que usa la cátedra, o `create_react_agent`) y qué preguntas pasan por ahí |
| Interpretar imágenes enviadas por el cliente | ⏳ | Prometido en el mail al profesor. Modelo verificado: `qwen/qwen3.8-27b` (~1.300-1.800 tokens por foto). Falta decidir si describe la foto para el dictamen actual o si dictamina él |
| Poder cambiar de modelo desde la configuración | ⏳ | Conviene antes de las imágenes, que usan un segundo modelo |
| Mostrar en la web de dónde sale cada respuesta (fragmentos del manual, herramientas, dictamen) | ⏳ | Luce mucho en el video |
| Más autos de prueba en la base | ⏳ A confirmar | Hoy el único auto con GNC es también el único C+, así que las pruebas no separan el efecto del producto del efecto del GNC |
| Exclusiones por gas licuado (GLP) y por uso para transporte de pasajeros | ⏳ A confirmar | El manual las tiene; se sumarían con el mismo mecanismo que el GNC |
| Historial que sobreviva al cerrar la app (SQLite) | ⏳ Baja prioridad | Suma una dependencia |
| Ajustar el tamaño y el solapamiento de los fragmentos del manual | ⏳ Baja prioridad | Cualquier cambio obliga a repetir la prueba del GNC |

## Por decidir

- **Granizo o parabrisas en un auto con GNC.** El manual es ambiguo: la lista
  general de exclusiones incluye el GNC, pero con la salvedad "excepto que la
  cobertura incluya alguno de los puntos enumerados". Hoy el sistema responde
  "Cubierto"; la otra lectura defendible es "Requiere peritaje". Tiene que
  coincidir con lo que se diga en la defensa.

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
| Unificar `ONBOARDING.md` y `SETUP.md` | ⏳ |
| Paquete para el profesor con la menor fricción posible: key de Groq de una cuenta nueva dedicada a la entrega (con 1-2 meses de vida, y revocarla después), instalación desde `requirements-lock.txt`, un script de instalación de un solo comando y la base vectorial ya armada | ⏳ |
| Probar la instalación desde cero siguiendo solo las instrucciones escritas | ⏳ |

---

El análisis original de Nicolás, con el detalle de cada mejora propuesta
(qué falta, por qué importa y dónde tocar), está en el historial:
`git show 8691493:PENDIENTES.md`.
