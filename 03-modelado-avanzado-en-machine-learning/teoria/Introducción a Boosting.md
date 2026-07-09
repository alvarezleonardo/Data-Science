<!-- Convertido automáticamente desde: Introducción a Boosting.pdf -->

Introducción a Boosting

01 Boosting

La principal diferencia con las técnicas de Bagging es que en ellas se entrenaban
los modelos independientemente para luego generar un meta-modelo. En el caso de
Boosting, se entrenan los modelos de manera **secuencial**, donde cada modelo
aprende de los errores del modelo predecesor.

Gradient Boost

Gradient Boosting es un método de aprendizaje **lento** donde los sucesivos
modelos de árboles de decisión son entrenados para predecir los **residuales**
del árbol antecesor.

Hiperparámetro de Learning Rate (η):
- El Learning Rate (η) es un escalar entre 0 y 1 (`0 < η < 1`) que multiplica los
  residuales para asegurar la convergencia.
- Recomendación: a medida que se reduce el valor de η, es recomendable aumentar
  el número de estimadores (N).

La predicción del meta-modelo:

    ŷ_pred = y₁ + η·r₁ + η·r₂ + … + η·r_N

donde cada árbol se entrena sobre los residuales del anterior:
`r₁ = y₁ − ŷ₁`, `r₂ = r₁ − r̂₁`, `r₃ = r₂ − r̂₂`, …

Implementación en sklearn:
- Regresión: `GradientBoostingRegressor`
- Clasificación: `GradientBoostingClassifier`

ADA Boost

ADA proviene de Adaptative Boosting, que hace referencia a su capacidad de
adaptar la importancia de los predictores, asignando mayor peso a aquellos sobre
los que se comete más error.

Comparación con Random Forest:
- Poda de árboles: en Random Forest los árboles no se podan; en ADA Boost se usan
  árboles de 1 nodo raíz y 2 nodos hoja, conocidos como **stump**.
- Peso en la predicción: en Random Forest cada árbol tiene igual voto; en ADA
  Boost los votos de los stumps pueden tener distintos pesos.

Funcionamiento de ADA Boosting:
- Creación del primer stump:
  1. Selección del feature: se elige el que genera la menor entropía o índice de
     Gini y se crea el primer stump.
  2. Cálculo de la importancia del árbol: según la cantidad de error cometido. El
     error se calcula sumando los pesos de las observaciones mal clasificadas
     (valor entre 0 y 1). Error = 1 → no clasificó nada bien (poco voto); Error =
     0 → clasificó todo (mayor voto).
  3. Ajuste de pesos: se ajustan los pesos de cada observación para que el
     siguiente stump aprenda del anterior. La suma total de los pesos es 1.
- Creación de los siguientes stumps:
  4. Selección de partición: el segundo stump usa el **weighted Gini index**.
  5. Repetición: se calcula el error total del stump para asignar su peso, y se
     recalculan los pesos de cada observación para el siguiente.

Predicción del meta-modelo en ADA Boosting:
- Clasificación: voto con pesos → `AdaBoostClassifier`.
- Regresión: promedio ponderado → `AdaBoostRegressor`.

XG Boost

XGBoost (Extreme Gradient Boosting) es una técnica avanzada de boosting que
calcula los gradientes de segundo orden (las segundas derivadas parciales de la
función de pérdida).

- Gradientes de segundo orden: proporcionan más información sobre la dirección de
  los gradientes y cómo llegar al mínimo de la función de pérdida. Mientras el
  gradient boosting usa la función de pérdida del modelo base como proxy, XGBoost
  usa la derivada de segundo orden como aproximación.
- Ventajas del entrenamiento: velocidad (muy rápido) y paralelización (se puede
  distribuir entre clústeres).
