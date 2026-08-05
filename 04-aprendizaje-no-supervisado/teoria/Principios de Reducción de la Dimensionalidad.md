# Principios de Reducción de la Dimensionalidad

> Conversión a Markdown de la slide del curso (Clase 22, Módulo 5). El PDF original está al lado.

La **reducción de dimensionalidad** es una técnica esencial en el análisis de datos: busca **disminuir el número de variables (dimensiones)** de un conjunto de datos **preservando la mayor cantidad de información relevante** posible. Mejora la eficiencia de los algoritmos, facilita la interpretación de los datos y optimiza la calidad de los modelos, especialmente en contextos de **alta dimensión**.

> Idea visual (3D vs 2D): un conjunto de datos en 3 dimensiones puede proyectarse a 2 dimensiones manteniendo la estructura general (los grupos siguen separados).

## 1. Importancia y fundamentos

- **Visualización:** en datos con muchas dimensiones, interpretar y visualizar es complicado. Reducir a **2 o 3 dimensiones** permite crear gráficos más sencillos y detectar patrones.
- **Mejora del rendimiento del modelo:** eliminar variables irrelevantes o redundantes **reduce el ruido** → mejora la precisión (ej. datasets de imágenes).
- **Simplificación del modelo:** un modelo más simple **generaliza mejor** a datos nuevos y **evita el sobreajuste** (menos features → menos riesgo de memorizar el train).
- **Mitiga la maldición de la dimensión:** en alta dimensión los datos se dispersan y las distancias se vuelven poco significativas; reducir dimensiones mejora los algoritmos basados en distancias (K-means, K-NN). Ver [Comprender la Maldición de la Dimensión](<Comprender la Maldición de la Dimensión.md>).

## 2. Métodos comunes de reducción de dimensionalidad

| Método | Tipo | Idea | Uso típico |
|--------|------|------|-----------|
| **PCA** | Lineal, no supervisado | Nuevas dimensiones ordenadas por **varianza**; los primeros componentes capturan la mayor parte | Reducir dimensionalidad conservando información |
| **t-SNE** | No lineal | Preserva la **estructura de vecinos locales** | **Visualización** de datos complejos en 2-3D |
| **ICA** | Lineal | Componentes **estadísticamente independientes** | Separar señales mezcladas (ej. fuentes de audio) |
| **Selección de características** | — | Elige un **subconjunto** de las variables originales (no las transforma) | Mantener interpretabilidad; árboles / Random Forest por importancia |

**Distinción clave:**
- **Extracción de características** (PCA, t-SNE, ICA, LDA): **crea nuevas** variables combinando las originales.
- **Selección de características** (filtros, embedded): **elige** un subconjunto de las variables originales sin transformarlas.

> Ver también la [Referencia Técnica del módulo](../REFERENCIA-TECNICA.md) para la comparativa PCA vs LDA vs t-SNE y cuándo usar cada una.
