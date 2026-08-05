# Estrategias de Selección de Variables

> Conversión a Markdown de la slide del curso (Clase 17, Módulo 4). El PDF original está al lado.

La **selección de variables** es fundamental en el análisis de datos y la creación de modelos: permite **reducir el número de características** y enfocarse en las que aportan mayor valor. Esto mejora el rendimiento del modelo, facilita su **interpretación**, reduce el riesgo de **sobreajuste** y acelera los tiempos de entrenamiento.

## 1. Métodos Basados en Filtros

Técnicas de selección de características que evalúan la importancia de las variables en función de **medidas estadísticas**, **sin utilizar un modelo** de aprendizaje automático. Se aplican **antes de entrenar** un modelo y ayudan a reducir la dimensionalidad seleccionando las características más relevantes.

### Método de la Correlación
- **Descripción:** evalúa la correlación entre cada característica y la variable objetivo. Se seleccionan las de **alta correlación** y se descartan las de baja.
- **Ejemplo:** en regresión, se usa el coeficiente de **Pearson**; las características con coeficiente más cercano a **1 o -1** se consideran más relevantes.

### Prueba de Chi-Cuadrado
- **Descripción:** usa la prueba estadística de **chi-cuadrado** para evaluar la **independencia** entre las características y la variable objetivo. Adecuado para **variables categóricas**; identifica las que tienen dependencia significativa con el target.
- **Ejemplo:** en clasificación, selecciona características con relación significativa con las clases. Mayor dependencia con la clase → más relevante (a mayor score, menor p-value).

### Análisis de Varianza (ANOVA)
- **Descripción:** compara la **varianza** de las características entre diferentes clases para identificar aquellas con **diferencias significativas en sus medias**. Útil para seleccionar características relevantes en clasificación.
- **Ejemplo:** ANOVA evalúa qué características tienen variabilidad significativa entre clases (F-score), ayudando a elegir las que mejor **separan las categorías**.

### Coeficientes de Importancia (para Modelos Lineales)
- **Descripción:** en modelos lineales (regresión lineal o logística) la importancia se mide con los **coeficientes** del modelo. Mayor magnitud (en **valor absoluto**) → más relevante.
- **Ejemplo:** se priorizan las características con mayor valor absoluto de coeficiente, es decir, con mayor influencia sobre la variable objetivo.

## 2. Eliminación Recursiva de Características (RFE)

**RFE** es un método de selección que usa un **enfoque iterativo** para eliminar las características menos importantes, **basándose en un modelo** de aprendizaje automático. A diferencia de los métodos de filtros, RFE **emplea un modelo** para evaluar la importancia de las características en cada iteración.

### Etapas
1. **Inicialización:** se comienza con **todas** las características y se entrena un modelo (SVM, regresión, etc.).
2. **Evaluación de importancia:** se evalúa la importancia de cada característica con el modelo entrenado (coeficientes en modelos lineales, importancia en árboles).
3. **Eliminación:** se elimina la característica **menos importante** (menor coeficiente o menor contribución al rendimiento).
4. **Iteración:** se reentrena el modelo con las características restantes y se repite evaluación + eliminación, hasta alcanzar el número deseado o un criterio de parada.
5. **Selección final:** se elige el subconjunto de características que **maximiza el rendimiento** del modelo.

## 3. Métodos Basados en Filtros vs RFE

| Criterio | Métodos Basados en Filtros | RFE (Eliminación Recursiva) |
|----------|----------------------------|-----------------------------|
| **Independencia del modelo** | No dependen de un modelo; se basan en estadísticas generales de las características. | Dependen de un modelo de aprendizaje automático para evaluar la importancia. |
| **Computación** | Más rápidos, no requieren entrenar un modelo en cada paso. | Más costoso: implica entrenar el modelo **iterativamente**. |
| **Interpretabilidad** | Evaluación directa de las características, sin influencia de un modelo. | Evaluación basada en el rendimiento del modelo, más relevante en algunos contextos. |
| **Flexibilidad** | General, aplicable **antes** de la modelización. | Centrado en el rendimiento del modelo, mejora la **precisión** del modelo final. |

> Enlaza con [Comprender la Maldición de la Dimensión](<Comprender la Maldición de la Dimensión.md>) (Clase 16) y anticipa la selección con **regularización** (Lasso/Ridge, Clase 19) y métodos **embedded** (árboles, Clase 20).
