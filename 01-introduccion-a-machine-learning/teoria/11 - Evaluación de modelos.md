<!-- Convertido automáticamente desde: 11 - Evaluación de modelos.pdf -->

Evaluación de
Modelos

La Validación cruzada o cross validation  es un
método que consiste en evaluar y probar el
rendimiento de un modelo de machine learning,
con el ﬁn de encontrar un mejor modelo
rápidamente. Esta técnica ayuda a la comprensión
y aplicación de este modelado predictivo, siendo
fácil y sencilla de aplicar.

La idea de la validación cruzada es simple: en lugar
de usar diferentes sets para entrenamiento y
validación ¡en la validación cruzada usaremos
todos los datos!

Cross Validation

Validación Cruzada

La validación cruzada es una técnica utilizada para evaluar la capacidad predictiva de un modelo de
machine learning, asegurando que este se generalice bien a datos no vistos. Este método implica dividir
el dataset original en múltiples subconjuntos, conocidos como "folds". El proceso típico sigue estos
pasos:

//

● División del Dataset: Se divide el dataset en k subconjuntos de aproximadamente el mismo

tamaño. Este valor k se elige según la necesidad del análisis, siendo común el uso de k = 5 o k = 10

● Entrenamiento y Evaluación:

○ Para cada uno de los k subconjuntos, el modelo se entrena usando 𝑘−1 de esos subconjuntos

como datos de entrenamiento.

○ El subconjunto restante se usa como conjunto de prueba.
○ Este proceso se repite k veces, cambiando el subconjunto de prueba en cada iteración.

● Promedio de Resultados: Se calculan los resultados de evaluación (como precisión, error, etc.) en
cada una de las k iteraciones. Luego, se promedian estos resultados para obtener una medida ﬁnal
del desempeño del modelo.

Uso Eﬁciente de Datos: Maximiza el uso de los
datos disponibles, ya que cada dato se utiliza tanto
para entrenamiento como para prueba.

Evaluación Robusta: Proporciona una estimación
más ﬁable del rendimiento del modelo, ya que se
evalúa en múltiples particiones del dataset.

Ventaja de la
Validación
Cruzada

Tipos comunes de Validación Cruzada

● k-Fold Cross Validation: Como se describió, el dataset se divide en k partes.
● Leave-One-Out Cross Validation (LOOCV): Es un caso extremo donde k es igual al número
de observaciones en el dataset. Cada iteración usa una sola observación como conjunto de
prueba y el resto como conjunto de entrenamiento.

● Stratiﬁed k-Fold Cross Validation: Similar a k-Fold, pero se asegura de que cada fold

mantenga la proporción original de clases en problemas de clasiﬁcación, proporcionando así
una evaluación más equilibrada.

//

Los hiperparámetros son las características externas de un
modelo, no se "aprenden", son valores que tiene que
deﬁnirse cuando se implementa el modelo, antes de
entrenarlo.
Un ejemplo es el valor k en el algoritmo K-means, o el nivel
máximo de un árbol.

Los parámetros son las características internas de un
modelo, son valores estimados a partir del entrenamiento
con los datos.
Un ejemplo son los coeﬁcientes de una regresión lineal.

Los parámetros son las características internas de un
modelo, son valores estimados a partir del entrenamiento
con los datos.
Un ejemplo son los coeﬁcientes de una regresión lineal.

Hiperparámetros

Hiperparámetros

No podemos saber a priori cuáles son los mejores valores de
los hiperparámetros del modelo para resolver un problema
determinado. Además, mientras una conﬁguración
especíﬁca de hiperparámetros puede generar una buena
performance del modelo para un determinado dataset, en
otro similar quizás no sea así.

Por lo tanto, la optimización o ajuste de los
hiperparámetros (hyperparameters tuning), es decir, la
selección de un conjunto óptimo de valores para los
hiperparámetros, es una parte esencial del machine learning.
Sin embargo, los modelos pueden tener muchos
hiperparámetros, y encontrar la mejor combinación se
convierte en un problema de búsqueda de información.

//

Hiperparámetros

Vamos a analizar dos métodos de ajuste de hiperparámetros: Grid
Search y Random Search. Ambos métodos siguen un proceso simple:

Para cada combinación de valores de los hiperparámetros:

● Aplican los valores sobre el conjunto de entrenamiento.
● Evalúan con validación cruzada.
● Registran el puntaje obtenido.

Al ﬁnal de todas las búsquedas:

● Seleccionan la combinación con el puntaje más alto.
● Aplican la combinación seleccionada sobre el conjunto de

entrenamiento.

● Predicen sobre el conjunto de prueba.

//

GridSearch busca la mejor combinación de hiperparámetros
dentro de una grilla (grid) especiﬁcada previamente. La
búsqueda es exhaustiva para cada valor de la grilla.

RandomSearch selecciona en forma aleatoria un subset de los
hiperparámetros

Es importante tener en cuenta el costo a nivel de cómputo de
evaluar todas las combinaciones de los valores de los
hiperparámetros. Por lo tanto, en algunos casos vamos a tener
que elegir una grilla reducida, o usar RandomSearch que
selecciona un subset de combinaciones.

Hyperparameter
Tuning

Cómo se implementa

Para implementar una búsqueda de hiperparámetros en sklearn, debemos considerar los
siguientes elementos:

1. Un estimador: Es decir, un modelo sobre el cual queremos trabajar.
2. Un espacio de parámetros: El rango de valores de los hiperparámetros sobre los

cuales se realizará la búsqueda.

3. Un método de búsqueda: La estrategia utilizada para buscar entre los modelos

candidatos (por ejemplo, RandomSearch o GridSearch).

4. Un esquema de validación cruzada: Seleccionando la cantidad de folds para dividir

el dataset.

5. La métrica de evaluación: El criterio utilizado para elegir el mejor modelo.

//

¡Muchas gracias!


