# Idea de Sistema 2: Agente conversacional para app de seguros

&nbsp;

### **1\. El componente RAG (Consultas sobre la Póliza)**

Los seguros están llenos de "letra chica" (condiciones generales, exclusiones, límites de franquicia). Nadie lee esos PDFs.

* **¿Cómo se hace en el TP?** Cargarías documentos PDF reales (o simulados) de condiciones de pólizas de autos o de hogar usando el PyPDFLoader de LangChain. Los partís en *chunks*, generás los embeddings y los guardás en Chroma DB.  
* **Ejemplo de uso:** El usuario pregunta *"¿Me cubre si me rompen el vidrio lateral por un robo y tengo franquicia?"*. El sistema usa RAG, busca en la póliza específica, y le responde de manera clara y resumida.

### **2\. El componente Agente y Tools (Transacciones)**

Al igual que en el e-commerce, el agente necesita herramientas (@tool) para actuar, pero enfocadas en el rubro de seguros:

* **Tool consultar\_estado\_poliza(dni, patente):** Ejecuta una consulta (SQL simulado) a tu base de datos para decirle al usuario si está al día con los pagos o si su póliza está vencida\[cite: 3, 5\].  
* **Tool registrar\_siniestro(datos\_vehiculo, fecha, descripcion):** Si el usuario dice *"Choqué hoy a la mañana en Av. Pellegrini"*, el Agente extrae esa información y usa esta herramienta para hacer un INSERT en una tabla de SQLite llamada siniestros, dándole al usuario un número de reclamo al instante\[cite: 3, 5\].  
* **Tool pedir\_grua(ubicacion):** Una herramienta que simule despachar un servicio de asistencia mecánica.

# Stack Tecnológico Principal

---

| Componente / Rol | Tecnología Seleccionada | Descripción de Uso |
| :---- | :---- | :---- |
| Orquestación y Ciclo de Vida | LangChain y LangGraph | Gestión del flujo ReAct, memoria del usuario y definición de Tools. |
| Motor de Razonamiento (LLM) | GPT (API OpenAI) o LLaMA (Local) | Toma de decisiones, interpretación de lenguaje natural y generación de respuestas. |
| Base de Datos Vectorial | Chroma DB / FAISS | Almacenamiento y recuperación rápida de embeddings del catálogo. |
| Representación Semántica | Embeddings (OpenAI / Gemini) | Traducción de los productos de la tienda a espacios vectoriales (RAG). |

&nbsp;

# Diseño Conceptual

1. El usuario envía un mensaje desde la Interfaz.  
2. El Orquestador (LangChain) recupera el historial de la Memoria y le pasa todo el paquete al Motor de Razonamiento (LLM). LangChain y LangGraph se encargan de la gestión del flujo ReAct, la memoria del usuario y la definición de Tools.  
3. El LLM **alterna entre razonar una pregunta y decidir utilizar una herramienta específica** (como la herramienta buscar\_productos(descripcion) que se conecta al módulo RAG (embeddings en base vectorial), o una consulta SQL simulada a la base de datos). Para la memoria y mantener el contexto de la conversacion usamos Runnable With Message History o los grafos de estado de LangGraph.  
4. El sistema ejecuta la herramienta y le devuelve el dato crudo al LLM. El LLM se encarga de **observar el resultado devuelto**.&nbsp;  
5. Finalmente, con esa información, el LLM procede a **construir la respuesta** y se la envía de vuelta al usuario.  
   &nbsp;

### **1\. Interfaz de Usuario (El Frontend)**

Es el punto de entrada donde el usuario escribe su consulta. Para que la experiencia de chat sea fluida, se puede implementar WebSocket para que el chat sea en tiempo real.

### **2\. Orquestador Central: LangChain y LangGraph**

Este es el "cerebro" integrador del sistema. LangChain es el marco de trabajo obligatorio diseñado para encadenar componentes de IA, utilizando LCEL (LangChain Expression Language) para conectar fácilmente plantillas, modelos y herramientas.&nbsp;

### **3\. Sistema de Memoria e Historial**

Para que el agente no pierda el hilo de la conversación, este módulo se conecta al Orquestador. Se logra mediante herramientas como Runnable With Message History o los grafos de estado de LangGraph, permitiendo que el sistema mantenga un historial conversacional y entienda el contexto previo del usuario.

### **4\. Motor de Razonamiento (LLM)**

El Orquestador envía el prompt, el mensaje del usuario y el historial a este motor, que puede ser GPT (vía API de OpenAI) o un modelo local como Ollama. Su función es la toma de decisiones, la interpretación del lenguaje natural y la generación de las respuestas.

### **5\. Módulo RAG (Base de Conocimiento)**

Se utiliza para que el LLM responda consultando información externa (como el catálogo de productos o la "letra chica" de las pólizas). Funciona así:

* **Embeddings:** Se utilizan modelos de OpenAI o Gemini para transformar los textos en Embeddings, que son vectores que logran capturar el significado semántico profundo de los documentos.  
* **Base Vectorial:** Estos vectores se almacenan en Chroma DB o FAISS para permitir un almacenamiento y recuperación rápida.  
* **Comunicación:** El sistema busca pasajes relevantes en este índice vectorial y luego el LLM genera la respuesta basándose en ese contexto recuperado.

### **6\. Herramientas Transaccionales (Mock de Base de Datos)**

Aquí es donde el agente cobra vida y ejecuta acciones. Las herramientas deben estar decoradas con @tool y se programan y validan localmente.

* Dado que no habrá tiempo de programar un sistema externo real completo, se creará una base de datos SQLite falsa (o diccionarios en Python) con tablas de productos, pedidos o usuarios.  
* Las herramientas interactuarán con esa base de datos. Ejemplos de esto son agregar\_a\_carrito(id\_producto, cantidad) o registrar\_siniestro(datos\_vehiculo, fecha, descripcion), donde esta última hace un INSERT en una tabla de SQLite.

&nbsp;