<!-- Convertido automáticamente desde: 12 - XG Boost.pdf -->

XGBoost

¿Qué es XGBoost?

XGBoost, abreviatura de eXtreme Gradient Boosting, es una biblioteca de machine
learning optimizada y distribuida que implementa el algoritmo de Gradient Boosting de
forma más eﬁciente y escalable. Fue desarrollada por Tianqi Chen y ha ganado
popularidad rápidamente, especialmente en competencias de datos como Kaggle.

//

Calcula gradientes de segundo orden, es decir, segundas
derivadas parciales de la función de pérdida, que proporciona
más información sobre la dirección de los gradientes y cómo
llegar al mínimo de nuestra función de pérdida. Mientras que
gradient boosting usa la función de pérdida de nuestro modelo
base (por ejemplo, árbol de decisión) como un proxy para
minimizar el error del modelo general, XGBoost usa la derivada
de segundo orden como una aproximación.

El entrenamiento es muy rápido y se puede paralelizar /
distribuir entre clústeres.

XGBoost

Características de XGBoost:

Eﬁciencia y Velocidad:

//

● Optimización del Uso de Recursos: XGBoost está diseñado para hacer un uso
eﬁciente de la memoria y la CPU, permitiendo el manejo de datasets grandes y
complejos.

● Paralelización: Soporta la paralelización en el entrenamiento de modelos, lo que

acelera signiﬁcativamente el proceso.

Manejo de Valores Perdidos:

● Valores Faltantes: XGBoost puede manejar automáticamente valores faltantes,
tratando de aprender la mejor dirección a seguir para la división de los datos.

Características de XGBoost:

Flexibilidad:

● Función de Costo Personalizable: Permite deﬁnir funciones de costo personalizadas según

las necesidades especíﬁcas del problema.

● Objetivos de Optimización: Soporta tanto objetivos de regresión como de clasiﬁcación, así

como objetivos personalizados.

Reducción de la Varianza:

● Boosting de Gradiente: Combina múltiples árboles de decisión débiles en un modelo fuerte,

reduciendo la varianza y mejorando la precisión del modelo.

Control del Crecimiento del Árbol:

● Pruning de Árboles: Implementa una estrategia de pruning (poda) para eliminar ramas no

signiﬁcativas, optimizando así la estructura del árbol.

//

Boosting es un procedimiento iterativo que va construyendo un
modelo ﬁnal en pasos. En cada nuevo paso intentará aprender de
los errores cometidos en los pasos previos. Se entrena una
secuencia de modelos donde se da más peso a los ejemplos que
fueron clasiﬁcados erróneamente por iteraciones anteriores.

Trabaja sobre los errores del modelo anterior o bien usándolos
para cambiar la ponderación en el siguiente modelo o bien
entrenando un modelo que prediga los mismos. Al igual que con
bagging, las tareas de clasiﬁcación se resuelven con una mayoría
ponderada de votos, y las tareas de regresión se resuelven con
una suma ponderada para producir la predicción ﬁnal.

Boosting

¡Muchas gracias!


