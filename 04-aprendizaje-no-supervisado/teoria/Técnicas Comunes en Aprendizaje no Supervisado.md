# Técnicas Comunes en Aprendizaje no Supervisado

> Conversión a Markdown de la slide del curso (Clase 5). El PDF original está al lado.

## 1. Principales técnicas utilizadas

### Clustering (agrupamiento)

- **K-Means:** agrupa los datos en un número **fijo de clusters (`k`)** minimizando la **varianza dentro de cada cluster**. Es eficiente, pero requiere especificar `k` de antemano.
- **Hierarchical Clustering (jerárquico):** construye un **árbol de clusters** combinando o dividiendo clusters de forma iterativa. **No requiere** especificar el número de clusters, pero puede ser **computacionalmente costoso**.
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise): agrupa basándose en la **densidad de puntos** e identifica clusters de **forma arbitraria**. Útil para clusters no esféricos y para **detectar ruido**.

### Reducción de dimensionalidad

- **PCA** (Principal Component Analysis): proyecta los datos a un espacio de **menor dimensión** reteniendo la **mayor varianza** posible. Útil para visualización y para preparar datos para otros algoritmos.
- **Factor Analysis:** similar a PCA, pero asume que las variables observadas están influenciadas por un número menor de **factores subyacentes**. Útil para modelar estructuras de correlación.
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): diseñado para **visualización en 2D/3D** de datos de alta dimensión, preservando la estructura **local y global**.
- **Autoencoders:** redes neuronales que aprenden una **representación compacta** (codificación) de los datos, permitiendo reconstruir los datos originales.

### Modelado de temas

- **LDA** (Latent Dirichlet Allocation): modelo **generativo** que asume que los documentos son mezclas de temas y los temas son mezclas de palabras. Descubre la estructura temática en grandes colecciones de texto.
- **NMF** (Non-Negative Matrix Factorization): factoriza una matriz de datos en dos matrices menores manteniendo la **no negatividad**; útil para temas y factores interpretables en texto e imágenes.

### Modelos de mezcla de gaussiana

- **GMM** (Gaussian Mixture Models): supone que los datos se generan a partir de una **mezcla de varias distribuciones gaussianas**. Útil para modelar datos con **subestructuras complejas**.

## 2. Comparación de los diferentes enfoques

> Cada técnica tiene sus propias ventajas y desventajas; la elección del enfoque adecuado depende del **tipo de datos**, el **objetivo del análisis** y los **recursos disponibles**.

### Clustering

**K-Means vs. DBSCAN:**
- **K-Means:** requiere el número de clusters como entrada, es sensible a la inicialización y no maneja bien clusters no esféricos ni el ruido.
- **DBSCAN:** no requiere número de clusters y puede identificar clusters de formas arbitrarias y detectar ruido, pero es sensible a los parámetros de densidad.

**K-Means vs. Clustering Jerárquico:**
- **K-Means:** más rápido y escalable para grandes conjuntos, pero necesita especificar `k`.
- **Jerárquico:** no requiere especificar `k`, pero es más lento y menos escalable.

### Reducción de dimensionalidad

**PCA vs. t-SNE:**
- **PCA:** bueno para reducir dimensionalidad reteniendo la **varianza global**; adecuado para preprocesamiento y modelos predictivos.
- **t-SNE:** mejor para **visualización 2D/3D** (preserva relaciones locales), pero lento y no apto para grandes conjuntos.

**PCA vs. Autoencoders:**
- **PCA:** lineal, eficiente y fácil de interpretar, pero no captura relaciones **no lineales**.
- **Autoencoders:** capturan relaciones no lineales y son flexibles, pero más complejos y requieren entrenamiento.

### Modelado de temas

**LDA vs. NMF:**
- **LDA:** basado en un modelo **probabilístico**; intuitivo para algunos problemas, pero más complejo de ajustar y entrenar.
- **NMF:** basado en factorización de matrices no negativas; más intuitivo para ciertos datos, pero menos flexible en algunos casos.

### Modelos de mezcla de gaussiana (GMM)

- **Versatilidad:** modela múltiples distribuciones gaussianas y encuentra clusters con diferentes formas y tamaños.
- **Complejidad:** requiere selección de parámetros y es más complejo de ajustar que métodos simples como K-Means.
