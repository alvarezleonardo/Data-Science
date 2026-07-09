<!-- Convertido automáticamente desde: Introducción a Random Forest.pdf -->

Implementación de Random Forest

01 Bagging

El problema inherente de los árboles de decisión es la alta varianza que poseen.
Si construimos un árbol sobre un dataset, lo dividimos en 2 partes y generamos un
árbol para cada una, lo más probable es que obtengamos 2 árboles muy disímiles
entre sí e incluso muy diferentes al árbol original. Los métodos de ensamble
ayudan a reducir el error por varianza, aumentando el accuracy final del modelo.

¿Qué es bagging?

El nombre Bagging proviene de Bootstrap Aggregation.

El método Bootstrap consiste en generar, a partir de un dataset original, N
nuevos datasets con la misma cantidad de variables independientes, tomando
muestras con repetición.
- Muestras con repetición: es posible tener observaciones repetidas dentro de los
  nuevos datasets.

Funcionamiento del método de ensamble (Bagging)

Generación de modelos:
- Genera N modelos distintos al entrenar N árboles de decisión con N datasets
  "distintos".
- Estos datasets son creados a partir de la técnica de bootstrapping.

Predicción final del meta-modelo:
- Clasificación: votación por mayoría (la clase indicada por la mayoría de los
  modelos).
- Regresión: promedio de las regresiones de los modelos.

Características del Bagging

- Poda de árboles: al realizar Bagging NO se realiza poda de los N árboles. Esto
  es para que tengan el menor error de bias posible, incluso si tienen un gran
  error de varianza.
- Reducción del error de varianza: mediante la agregación, si la varianza de los
  árboles es `S²`, se espera que el modelo de Bagging alcance una varianza de
  `S² / N`.

02 Random Forest

Un Random Forest (Bosque Aleatorio) es un algoritmo de aprendizaje supervisado
que se utiliza tanto para clasificación como para regresión. Fue desarrollado por
Leo Breiman y Adele Cutler. Se basa en la construcción de múltiples árboles de
decisión durante el entrenamiento y la agregación de sus predicciones para
mejorar la precisión y controlar el sobreajuste.

Construcción del bosque:
1. Se generan múltiples subconjuntos de datos mediante muestreo con reemplazo
   (bootstrap sampling).
2. Para cada subconjunto, se construye un árbol de decisión de manera
   independiente.
3. Durante la construcción de cada árbol, en cada nodo se selecciona
   aleatoriamente un subconjunto de características para determinar la mejor
   división.
4. Predicción (clasificación): votación mayoritaria entre todos los árboles.
5. Predicción (regresión): promedio de las predicciones de todos los árboles.

Bagging es un caso particular de Random Forest donde el tamaño de la muestra de
predictores `M` coincide con el tamaño de la población de predictores `P`.

Empíricamente, los siguientes valores de `M` suelen descorrelacionar los modelos:
- Para regresiones: `M = P/3`.
- Para clasificaciones: `M = sqrt(P)`. No obstante, si los predictores están
  altamente correlacionados, se pueden usar valores más pequeños de `M`.

La predicción del meta-modelo funciona igual que en Bagging.
