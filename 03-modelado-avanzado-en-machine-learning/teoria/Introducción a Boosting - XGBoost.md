<!-- Convertido automáticamente desde: Introducción a Boosting - XGBoost.pdf -->

XGBoost

¿Qué es XGBoost?

XGBoost, abreviatura de eXtreme Gradient Boosting, es una biblioteca de machine
learning optimizada y distribuida que implementa el algoritmo de Gradient
Boosting de forma más eficiente y escalable. Fue desarrollada por Tianqi Chen y
ganó popularidad rápidamente, especialmente en competencias de datos como Kaggle.

- Calcula gradientes de segundo orden (segundas derivadas parciales de la función
  de pérdida), lo que da más información sobre la dirección de los gradientes y
  cómo llegar al mínimo. Mientras el gradient boosting usa la función de pérdida
  del modelo base como proxy, XGBoost usa la derivada de segundo orden como
  aproximación.
- El entrenamiento es muy rápido y se puede paralelizar / distribuir entre
  clústeres.

Características de XGBoost

Eficiencia y velocidad:
- Optimización del uso de recursos: uso eficiente de memoria y CPU, permitiendo
  datasets grandes y complejos.
- Paralelización: soporta paralelización en el entrenamiento, acelerando el
  proceso.

Manejo de valores perdidos:
- Valores faltantes: XGBoost puede manejarlos automáticamente, aprendiendo la
  mejor dirección a seguir para la división de los datos.

Flexibilidad:
- Función de costo personalizable: permite definir funciones de costo según las
  necesidades del problema.
- Objetivos de optimización: soporta objetivos de regresión, clasificación y
  personalizados.

Reducción de la varianza:
- Boosting de gradiente: combina múltiples árboles de decisión débiles en un
  modelo fuerte, reduciendo la varianza y mejorando la precisión.

Control del crecimiento del árbol:
- Pruning de árboles: implementa una estrategia de poda para eliminar ramas no
  significativas, optimizando la estructura del árbol.

Nota sobre Boosting: es un procedimiento iterativo que construye el modelo final
en pasos, aprendiendo de los errores de los pasos previos y dando más peso a los
ejemplos mal clasificados. Al igual que con bagging, la clasificación se resuelve
con una mayoría ponderada de votos y la regresión con una suma ponderada.
