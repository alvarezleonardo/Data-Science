# UMAP para Reducción de Dimensionalidad

> Conversión a Markdown de la slide del curso (Clase 27, Módulo 5). El PDF original está al lado.

**UMAP** (Uniform Manifold Approximation and Projection) es una técnica **moderna** de reducción de dimensionalidad que ganó popularidad por preservar **tanto la estructura local como la global** de los datos, con una **eficiencia computacional** notable. Fue introducida por Leland McInnes, John Healy y James Melville en **2018**.

## 1. Introducción

UMAP es un algoritmo **no lineal** que usa conceptos de **geometría algebraica y topología** para proyectar datos de alta dimensión a un espacio reducido, manteniendo estructura local y global.

- **A diferencia de PCA** (lineal): UMAP captura **relaciones no lineales**.
- **En comparación con t-SNE:** UMAP es **más rápido** y maneja mejor **grandes conjuntos de datos**; además preserva mejor la estructura global.

## 2. ¿Cómo funciona UMAP?

1. **Construcción del grafo de vecindad:** en el espacio de alta dimensión, cada punto se conecta con sus **vecinos más cercanos**. Este grafo representa las relaciones de proximidad.
2. **Transformación en un espacio de probabilidad:** modela la similitud entre puntos con una **distribución de probabilidad**, tanto en alta como en baja dimensión.
3. **Optimización y minimización de divergencia:** minimiza la divergencia entre ambas distribuciones mediante **descenso de gradiente**, encontrando la mejor proyección.
4. **Conservación de la estructura local y global:** mantiene la proximidad de puntos cercanos **y** las relaciones entre clusters.

## 3. Aplicaciones prácticas

- **Exploración de datos:** visualización de datos complejos en baja dimensión.
- **Detección de estructuras/clusters:** al proyectar a 2D/3D, revela cómo se agrupan los datos.
- **Embeddings de palabras y documentos:** ver cómo se agrupan semánticamente.
- **Datos biológicos (bioinformática):** explorar datos genómicos y transcriptómicos.
- **Embeddings de imágenes** y **clasificación/separación** de clases multiclase.

## 4. Ventajas y desventajas

| Ventajas | Desventajas |
|----------|-------------|
| Preserva estructura **global y local** | **Parámetros sensibles** (`n_neighbors`, tamaño del embedding) |
| **Eficiente y escalable** (más que t-SNE) | La **interpretación** de resultados puede requerir análisis adicional |
| **Flexible**: ajustable con `n_neighbors` para distintas escalas de estructura | |

> **Nota de instalación:** UMAP no viene con scikit-learn; se instala aparte (`pip install umap-learn`) y se usa como `import umap`. Comparación completa en [Comparación de Técnicas de Reducción de Dimensionalidad](<Comparación de Técnicas de Reducción de Dimensionalidad.md>). Práctica: [notebooks/umap_practica.ipynb](../notebooks/umap_practica.ipynb).
