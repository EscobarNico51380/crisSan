# Repaso del TP1 — Clasificador de frutas y verduras (Kaggle)

Para la defensa del TP2, donde el profesor hace **una o dos preguntas sobre el
TP1** "para evaluar la comprensión global de los contenidos". Es material interno
del grupo: no va en el zip de la entrega.

Los números y las decisiones son del TP1 de Lucho
(`TP_Kaggle_Clasificador_Frutas_ENTREGA.ipynb`). Si tu TP1 fue distinto, los
conceptos son los mismos: sumá abajo lo propio.

**Cómo usarlo:** leer primero el TP en un minuto, después las preguntas
probables. Los conceptos están en el medio para consultar.

---

## 1. El TP en un minuto

- **Problema:** clasificar fotos en 5 clases (`apple`, `banana`, `grapes`,
  `potato`, `tomato`). Competencia de Kaggle de la cátedra.
- **Datos:** **325 imágenes** de entrenamiento y **89** de test. Muy pocas.
- **Métrica:** **F1 ponderado** (*weighted*), no accuracy.
- **Punto de partida:** el notebook de la cátedra, una red convolucional (CNN)
  entrenada desde cero con imágenes de 64 px: **0,57**.
- **Resultado final:** **1,00** en el leaderboard público, **sin entrenar
  ninguna red**: cuatro modelos preentrenados, congelados, que describen cada
  imagen como un vector de números, y una **regresión logística** encima que
  aprende a separar las clases.

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Base | CNN de la cátedra, 64 px | 0,57 |
| exp00 | La misma CNN, en local | 0,71 · F1 interno 0,62 |
| exp01 | ResNet18 preentrenado, *fine-tuning*, 224 px, LR 1e-4 | 0,96 · F1 interno 0,96 |
| exp01b | Lo mismo con otra semilla (para medir el ruido) | 0,96: igual, pero cambiaron 3 predicciones |
| exp03 | ConvNeXt congelado + regresión logística + validación cruzada | 0,96 · CV 0,994 |
| exp05 | 4 modelos congelados, vectores concatenados + regresión logística | **1,00** |

**La idea central en una frase:** con 325 imágenes, entrenar millones de
parámetros no funciona; lo que funciona es reusar lo que un modelo ya aprendió
con millones de imágenes y entrenar solo una capa chiquita encima.

---

## 2. Preguntas probables, con respuesta corta

**¿Qué es una red neuronal y cómo aprende?**
Capas de neuronas: cada una hace una suma ponderada de sus entradas y le aplica
una función de activación. Aprende ajustando los pesos: se calcula la salida
(*forward*), se mide el error con una función de pérdida, y la
**retropropagación** calcula, con la **regla de la cadena**, cuánto contribuyó
cada peso al error. Después se corrige cada peso un paso en la dirección que
baja el error (descenso por gradiente), con un tamaño de paso que es la **tasa
de aprendizaje**.

**¿Por qué el perceptrón simple no alcanza?**
Solo separa clases con una recta (o un hiperplano): no resuelve problemas **no
linealmente separables**, como el XOR. Con capas ocultas y activaciones no
lineales, la red puede aprender fronteras curvas.

**¿Por qué la CNN desde cero dio tan mal?**
Por dos razones medidas:
1. **Pocos datos para muchos parámetros:** 2,19 millones de parámetros con 233
   imágenes de entrenamiento.
2. **Resolución:** a 64 px, sobre fotos de 1200×1000, se conserva el 0,34 % de
   los píxeles. Una manzana y un tomate eran dos manchas rojas. Al pasar a
   224 px, el F1 de la manzana subió de 0,31 a 1,00 sin tocar nada más.

**¿Era sobreajuste o subajuste?**
**Subajuste**, al revés de lo esperable: la precisión en entrenamiento (50 %)
quedó *por debajo* de la de validación (76 %), y la pérdida de entrenamiento
quedó clavada desde la época 5. El modelo no aprendía ni sus propios datos.
Con ResNet18 pasó lo contrario: 99 % en entrenamiento contra 91 % en
validación, **sobreajuste**.

**¿Qué es el *transfer learning*?**
Usar un modelo ya entrenado con otro conjunto grande (ImageNet: más de un
millón de imágenes) como punto de partida. Las primeras capas ya saben detectar
bordes, texturas y formas, que sirven para cualquier foto. Hay dos formas:
- ***Fine-tuning***: se sigue entrenando todo el modelo con los datos propios
  (exp01: 11,2 millones de parámetros; sobreajustó).
- **Extracción de características**: el modelo queda **congelado** y se usa
  solo para convertir cada imagen en un vector; se entrena únicamente un
  clasificador chico encima (la solución final).

**¿Por qué la tasa de aprendizaje bajó de 1e-3 a 1e-4 con el preentrenado?**
Con pesos aleatorios hacen falta pasos grandes. Con pesos preentrenados, que ya
son buenos, pasos grandes los destruyen antes de que el modelo aprenda las
clases nuevas.

**¿Por qué un modelo grande si hay pocos datos?**
La regla "pocos datos, modelo chico" vale para lo que se **entrena**. Un
modelo congelado no suma parámetros entrenables: aporta una mejor descripción
de cada imagen. Lo que se entrena es la regresión logística.

**¿Por qué F1 ponderado y no accuracy?**
Las clases no están balanceadas (78 uvas contra 53 manzanas). El F1 combina
precisión (de lo que dije "manzana", cuánto era manzana) y recall (de las
manzanas, cuántas encontré); el ponderado promedia el F1 de cada clase según
cuántos ejemplos tiene. Con clases desbalanceadas, la accuracy puede ser alta
aunque una clase chica salga siempre mal.

**¿Qué es la validación cruzada y por qué se usó?**
Se parten los datos en 5 grupos y se entrena 5 veces, dejando afuera un grupo
distinto cada vez: cada imagen se evalúa una vez con un modelo que no la vio.
Con 325 imágenes, un único conjunto de validación de 46 imágenes es muy
ruidoso (cada imagen vale 2,2 puntos de F1). **Estratificada** significa que
cada grupo conserva la proporción de clases. Con el modelo congelado es barata:
los vectores se calculan una sola vez y las 5 vueltas son 5 regresiones.

**¿Qué es el sobreajuste y cómo se combate?**
Aprender de memoria los datos de entrenamiento en vez del patrón general: va
muy bien en entrenamiento y peor en datos nuevos. Se combate con más datos,
*data augmentation*, *dropout*, *weight decay*, menos épocas o, como acá, con
menos parámetros entrenables.

**¿Qué es el *data augmentation*? ¿Sirvió?**
Generar variantes de cada imagen al entrenar (girar, recortar, cambiar el
color) para que el modelo no dependa de detalles irrelevantes. Con el modelo
congelado casi no tiene efecto, porque el modelo no aprende. Y dos
transformaciones del notebook original (pasar a escala de grises y cambiar el
tono) **destruían el color**, la señal que distingue a la papa: el modelo
predecía 6 papas donde había ~17.

**¿Por qué el 1,00 del leaderboard no significa "perfecto"?**
El leaderboard público usa el 30 % del test: **~27 imágenes**. Cada una vale
3,7 puntos. Se comprobó corriendo lo mismo con dos semillas: cambiaron 3
predicciones y el puntaje público fue idéntico, porque los 3 cambios cayeron
en la parte privada. El puntaje final se calcula con las ~62 privadas.

**¿Usar imágenes del test para elegir el modelo no es hacer trampa?**
Es la pregunta más delicada. Se miraron a ojo 7 imágenes del test que los
modelos no se ponían de acuerdo en clasificar, y se usaron **solo para elegir
entre configuraciones ya entrenadas**; nunca para entrenar ni para reportar el
rendimiento (eso se reporta con la validación cruzada). Está declarado en el
notebook, con su sesgo: al elegir con el test, la elección queda un poco
sesgada hacia ese test.

**¿Qué es el *domain shift* y cuál era acá?**
Que los datos de prueba vengan de una distribución distinta a los de
entrenamiento. La consigna decía que eran los fondos, pero se midió que el 70 %
del entrenamiento ya eran fotos reales. El *shift* real eran imágenes que **no
son fotos**: el logo de Apple, un dibujo animado de una papa y manzanas
renderizadas en 3D. Solo uno de los cuatro modelos (EVA02, 448 px) reconocía
el logo como manzana.

**¿Qué aprendiste que no sea técnico?**
Medir antes de suponer. Medir el ruido antes de atribuir una mejora. Cuando un
instrumento de medición se satura, cambiarlo. Y mirar los datos: el problema
de fondo se resolvió mirando 7 imágenes, no ajustando el modelo.

---

## 3. Los conceptos de la cátedra, y dónde aparecen en el TP

| Concepto | Qué es | En el TP1 |
|---|---|---|
| **Perceptrón (Rosenblatt)** | Una neurona: suma ponderada + umbral. Aprende corrigiendo los pesos cuando se equivoca | La regresión logística final es, en esencia, una capa de neuronas con activación *softmax* |
| **Limitación del perceptrón** | Solo separa clases linealmente (no resuelve XOR) | Funcionó con una capa lineal porque los vectores de los modelos preentrenados dejan las clases separables: el trabajo no lineal ya lo hicieron las capas congeladas |
| **Red multicapa (Widrow)** | Capas ocultas entre la entrada y la salida | La CNN de la cátedra: 3 bloques convolucionales + 2 capas densas |
| **Función de activación** | Introduce la no linealidad (ReLU, sigmoide, softmax) | ReLU en las capas ocultas; softmax en la salida, que da la probabilidad de cada clase |
| ***Forward propagation*** | Calcular la salida pasando la entrada capa por capa | La predicción de cada imagen |
| ***Backpropagation*** | Propagar el error hacia atrás con la regla de la cadena para ajustar cada peso | El entrenamiento de la CNN y del *fine-tuning* |
| **Tasa de aprendizaje** | Tamaño del paso al corregir los pesos | 1e-3 desde cero; 1e-4 con el preentrenado |
| **Épocas** | Pasadas completas por los datos de entrenamiento | 20. Demasiadas épocas llevan a sobreajustar |
| **Sobreajuste / subajuste** | Aprender de memoria / no aprender ni lo básico | Subajuste con la CNN; sobreajuste con el *fine-tuning* |

Otros conceptos del notebook que conviene saber explicar:

- **Convolución:** un filtro chico que recorre la imagen buscando un patrón
  (un borde, una textura). Las primeras capas detectan cosas simples; las
  profundas, formas complejas.
- ***Pooling*:** reduce el tamaño de la imagen quedándose con lo más fuerte de
  cada zona.
- ***Batch normalization*:** normaliza las salidas de una capa; estabiliza y
  acelera el entrenamiento.
- ***Dropout*:** apaga neuronas al azar al entrenar para que la red no dependa
  de unas pocas. Es regularización. (En la CNN original, junto con el
  *augmentation*, hacía que la precisión de entrenamiento se viera más baja
  que la de validación.)
- **Adam:** el optimizador; ajusta la tasa de aprendizaje de cada peso por
  separado.
- **Pérdida de entropía cruzada:** mide qué tan lejos está la probabilidad
  predicha de la clase correcta. Es la estándar para clasificación.
- ***Weight decay*:** penaliza los pesos grandes. Regularización.
- **`class_weight="balanced"`:** da más peso a los errores de las clases con
  menos ejemplos. Sin eso, la regresión predijo 0 manzanas y 41 uvas.
- **Normalización L2:** cada vector queda de largo 1, así se comparan
  direcciones y no magnitudes.
- **TTA (*test-time augmentation*):** describir cada imagen normal y espejada y
  promediar. Una manzana espejada sigue siendo una manzana; promediar cancela
  ruido.
- **Fuga de datos:** una celda del notebook original volvía a copiar las
  imágenes de validación y test dentro del entrenamiento. La validación medía
  sobre fotos que el modelo ya había visto: los números no significaban nada.
  Se borró.
- **Guardar el mejor modelo según la métrica real:** el original guardaba el
  modelo con menor pérdida de validación, no con mejor F1; en exp00 se quedó
  con uno 4,4 puntos peor en precisión.

---

## 4. Dos inconsistencias del notebook entregado (tenerlas preparadas)

1. **Parámetros entrenables.** El resumen y las conclusiones dicen **3.845**
   (~12 por imagen), pero la salida del propio notebook dice **19.205**. El
   3.845 corresponde a un solo modelo (768 × 5 + 5); con los cuatro
   concatenados son 3.840 × 5 + 5 = **19.205**, ~59 por imagen. Si lo
   preguntan: fue un error al redactar, y la conclusión no cambia (sigue
   siendo ~580 veces menos que los 11,2 millones del *fine-tuning*).
2. **Imágenes disputadas.** El texto habla de **7** y la celda muestra **1**.
   Las 7 salieron de comparar los experimentos anteriores entre sí (exp01
   contra exp03); con los cuatro modelos finales, solo discrepan en una (el
   logo de Apple).

---

## 5. El puente con el TP2 (sirve para "la comprensión global")

La misma idea atraviesa los dos trabajos: **reusar un modelo preentrenado y
decidir con cuidado qué se le pide a cada pieza.**

| TP1 | TP2 |
|---|---|
| Modelos de visión preentrenados, congelados, convierten cada imagen en un vector | El modelo de *embeddings* (e5), preentrenado y congelado, convierte cada fragmento del manual en un vector |
| Se clasifica comparando vectores | Se busca comparando vectores (similitud coseno), más BM25 |
| Lo que se entrena es mínimo (una regresión logística) | No se entrena nada: el LLM se usa como viene y se le da el contexto (RAG) |
| Medir antes de suponer: el problema no eran los fondos | Medir antes de suponer: el GNC fallaba en la búsqueda, no en el modelo |
| El leaderboard de 27 imágenes no alcanzaba para decidir | El dictamen varía entre corridas: se prueba varias veces antes de concluir |
| Una métrica que se puede calcular (F1) la calcula el código | Lo que se puede calcular (fechas, vigencia) lo decide el código, no el LLM |

---

## 6. Lo propio de cada uno

Si tu TP1 fue distinto, anotá acá en tres líneas: qué modelo usaste, qué
puntaje sacaste y qué decisión te costó más.

- *(nombre)*:
