# Comparación de Técnicas de Reducción de Dimensionalidad

> Conversión a Markdown de la slide del curso (Clase 28, Módulo 5). El PDF original está al lado.

Cierre del módulo: comparación de las cuatro técnicas de reducción de dimensionalidad vistas (**PCA, LDA, t-SNE, UMAP**), con ventajas, desventajas y cuándo usar cada una.

## 1. PCA (Análisis de Componentes Principales)

- **Ventajas:** simple y **computacionalmente eficiente** (bueno en datasets grandes); captura la **varianza máxima** en un espacio lineal; componentes **interpretables** en términos de las variables originales.
- **Desventajas:** **lineal** (limitado para estructuras no lineales); **no considera las clases** (no optimiza la separación entre clases).
- **Casos de uso:** preprocesamiento antes de ML, visualización inicial de datos con estructura lineal, **compresión de datos**.

## 2. LDA (Análisis Discriminante Lineal)

- **Ventajas:** **maximiza la separación entre clases** (mejora la clasificación); reduce dimensionalidad manteniendo la estructura de clase (útil en multiclase).
- **Desventajas:** asume **normalidad** y matrices de varianza-covarianza iguales entre clases (no siempre se cumple); **lineal**.
- **Casos de uso:** **clasificación supervisada**, análisis de datos **con etiquetas** disponibles e importantes.

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

- **Ventajas:** excelente **preservación de la estructura local** (agrupa puntos similares); muy efectivo para **visualizar** datos complejos en 2D/3D.
- **Desventajas:** **escalabilidad** (costoso en datasets grandes); **preservación global limitada** (las distancias entre clusters pueden no ser fiables).
- **Casos de uso:** visualización de datos complejos, exploración de clusters.

## 4. UMAP (Uniform Manifold Approximation and Projection)

- **Ventajas:** preserva estructura **global y local**; **eficiente y escalable** (más que t-SNE); **flexible** (ajustable con `n_neighbors`).
- **Desventajas:** **dependencia de parámetros** (n_neighbors, dimensión objetivo); interpretación puede requerir análisis adicional.
- **Casos de uso:** visualización de datos de **alta dimensionalidad**, exploración y análisis de datos.

## Tabla comparativa

| Técnica | Lineal | Supervisada | Ventajas | Desventajas | Casos de uso |
|---------|:------:|:-----------:|----------|-------------|--------------|
| **PCA** | Sí | No | Simple, eficiente, interpretable | Lineal; ignora clases | Preprocesamiento, visualización inicial, compresión |
| **LDA** | Sí | **Sí** | Mejora separación de clases; mantiene estructura de clase | Asume normalidad; lineal | Clasificación supervisada, datos con etiquetas |
| **t-SNE** | No | No | Preserva estructura local; buena visualización | Escalabilidad; preservación global limitada | Visualización de datos complejos, exploración de clusters |
| **UMAP** | No | No | Preserva estructura global y local; eficiente; flexible | Depende de parámetros; interpretación compleja | Visualización de alta dimensión, exploración y análisis |

**Regla práctica de elección:**
- ¿Preprocesar / comprimir conservando varianza? → **PCA**.
- ¿Tengo etiquetas y quiero separar clases? → **LDA**.
- ¿Solo visualizar estructura local en un dataset chico/mediano? → **t-SNE**.
- ¿Visualizar dataset grande preservando estructura local y global? → **UMAP**.

> Enlaza con [PCA](<Análisis de Componentes Principales (PCA).md>), [LDA](<Análisis Discriminante Lineal (LDA).md>), [t-SNE](<T-SNE para Visualización.md>) y [UMAP](<UMAP para Reducción de Dimensionalidad.md>).
