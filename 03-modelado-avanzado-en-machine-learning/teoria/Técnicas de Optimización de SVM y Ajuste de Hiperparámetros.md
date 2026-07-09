<!-- Convertido automáticamente desde: Técnicas de Optimización de SVM y Ajuste de Hiperparámetros.pdf -->

Técnicas de Optimización de SVM y Ajuste de Hiperparámetros

01 Introducción a la Optimización en SVM

La optimización en SVM implica ajustar los parámetros del modelo para maximizar su
rendimiento en datos no vistos. Un buen ajuste de los hiperparámetros puede
mejorar significativamente la precisión y generalización del modelo. Hay que
encontrar el balance entre el ajuste excesivo (**overfitting**) y el ajuste
insuficiente (**underfitting**), tanto en clasificación como en regresión.

Hiperparámetros en SVM

- **C:** controla la penalización por errores de clasificación. Un valor alto de
  `C` trata de clasificar correctamente todos los puntos de entrenamiento,
  potencialmente llevando a overfitting.
- **Gamma (γ) en Kernel RBF:** define la influencia de un solo ejemplo de
  entrenamiento. Un valor alto de gamma significa una alta variabilidad en la
  función de decisión.
- **Kernel:** tipos comunes (lineal, polinomial, RBF, sigmoidal). La elección del
  kernel puede cambiar drásticamente el rendimiento del modelo.

02 Métodos de Búsqueda de Hiperparámetros

Grid Search: realiza una búsqueda **exhaustiva** sobre un conjunto especificado de
valores de hiperparámetros (todas las combinaciones de la grilla).
- Ventajas: exhaustividad (explora todas las combinaciones); simplicidad (fácil de
  implementar y entender).
- Desventajas: costo computacional muy alto con muchos hiperparámetros o rangos
  amplios; no escala bien.
- sklearn: `GridSearchCV`.

Random Search: realiza una búsqueda **aleatoria** sobre el espacio de
hiperparámetros.
- Ventajas: eficiencia (suele encontrar buenos resultados más rápido que Grid
  Search, explorando más variedad en menos tiempo); flexibilidad (no requiere un
  número fijo de combinaciones; puede detenerse al alcanzar un buen resultado).
- Desventajas: incertidumbre (no garantiza la combinación óptima); depende de cómo
  se defina el espacio de búsqueda.
- sklearn: `RandomizedSearchCV`.

Bayesian Optimization: modelo **probabilístico** que usa la información de
iteraciones previas para decidir las próximas combinaciones a evaluar, enfocando
la búsqueda en áreas prometedoras.
- Ventajas: eficiencia en recursos (reduce el número de evaluaciones); optimización
  dirigida.
- Desventajas: más complejo de implementar y entender; puede requerir más tiempo en
  las primeras iteraciones para construir el modelo probabilístico (se compensa
  luego).
