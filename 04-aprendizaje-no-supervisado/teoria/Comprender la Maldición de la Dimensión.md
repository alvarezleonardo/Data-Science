# Comprender la Maldición de la Dimensión

> Conversión a Markdown de la slide del curso (Clase 16, Módulo 4). El PDF original está al lado.

## 1. Definición

La **maldición de la dimensionalidad** (curse of dimensionality) se refiere a los problemas que surgen al trabajar con datos en **espacios de alta dimensionalidad**. A medida que el número de dimensiones aumenta, los datos se **dispersan**, lo que reduce la efectividad de varios algoritmos —especialmente los **basados en distancias**— afectando su **precisión y generalización**.

## 2. Efectos de la maldición de la dimensión

- **Dispersión y distancias poco significativas:** en alta dimensión los datos están más dispersos; las distancias entre puntos se vuelven **más uniformes** y menos significativas. La diferencia entre puntos cercanos y lejanos disminuye → caen en efectividad algoritmos como **K-means** y **K-NN**.
- **Complejidad computacional:** crece **exponencialmente** con el número de dimensiones; tiempo y memoria aumentan mucho, volviendo impracticables algunos algoritmos.
- **Overfitting:** en alta dimensión los modelos tienden a **aprender el ruido** en lugar de la estructura subyacente, ajustándose demasiado a train y perdiendo rendimiento en datos no vistos.
- **Reducción de la calidad de los resultados:** por la dispersión y el ruido, clustering y clasificación se vuelven **menos confiables**.
- **Problemas de visualización:** visualizar más de 3 dimensiones es complicado; dificulta interpretar patrones y relaciones.

## 3. Desafíos y problemas comunes

- **Selección de características:** elegir las dimensiones relevantes es difícil; más features generan un **espacio de búsqueda** más grande y dificultan identificar las más relevantes.
- **Curse of dimensionality en el clustering:** en K-means, muchas dimensiones hacen que los clusters sean **menos claros** (los métodos por distancia pierden efectividad por la uniformidad de distancias).
- **Costo computacional y eficiencia algorítmica:** más dimensiones → más procesamiento/almacenamiento (mayor costo); algoritmos que rinden bien en baja dimensión pueden **no escalar** y degradar su eficiencia.

## 4. Estrategias para mitigar la maldición de la dimensión

- **Reducción de dimensionalidad:** aplicar **PCA, t-SNE o UMAP** para reducir el número de dimensiones manteniendo la estructura importante.
- **Selección de características:** métodos basados en la **importancia** de features o técnicas de filtro para retener solo las más relevantes.
- **Regularización:** técnicas **L1 / L2** para reducir el riesgo de overfitting y mejorar la generalización.
- **Algoritmos eficientes para alta dimensionalidad:** usar algoritmos diseñados para ello (basados en árboles o técnicas de clustering especializadas).
- **Ajuste de parámetros:** optimizar los parámetros de los algoritmos para mejorar su rendimiento en espacios de alta dimensión.

> Enlaza con el Módulo 4 del curso (Selección de Variables) y el Módulo 5 (Reducción de la Dimensionalidad: PCA, LDA, t-SNE, UMAP).
