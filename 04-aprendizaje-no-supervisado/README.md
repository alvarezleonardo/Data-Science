# 04 — Aprendizaje no supervisado

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 6 | 31 | 🔄 En curso (65% · 20/31 clases) |

Técnicas de aprendizaje no supervisado: clustering (K-Means, jerárquico, DBSCAN), reducción de dimensionalidad (PCA, LDA, t-SNE, UMAP) y selección de variables. Curso "ML3" del programa de Digital House.

> **[REFERENCIA-TECNICA.md](REFERENCIA-TECNICA.md)** — manual de consulta rápida con los temas clave del módulo: qué es cada técnica, cuándo aplicarla, para qué, hiperparámetros, cómo se evalúa y snippets de `scikit-learn`. Pensado para repaso de examen.

## Contenido

### `teoria/`
Slides del curso (PDF + conversión `.md`):

| Clase | Tema | PDF | Markdown |
|:-----:|------|-----|----------|
| 3 | Principios del Aprendizaje no Supervisado | [PDF](<teoria/Principios del Aprendizaje no Supervisado.pdf>) | [MD](<teoria/Principios del Aprendizaje no Supervisado.md>) |
| 4 | Aplicaciones del Aprendizaje no Supervisado | [PDF](<teoria/Aplicaciones del Aprendizaje no Supervisado.pdf>) | [MD](<teoria/Aplicaciones del Aprendizaje no Supervisado.md>) |
| 5 | Técnicas Comunes en Aprendizaje no Supervisado | [PDF](<teoria/Técnicas Comunes en Aprendizaje no Supervisado.pdf>) | [MD](<teoria/Técnicas Comunes en Aprendizaje no Supervisado.md>) |
| 6 | Preparación de Datos para Aprendizaje no Supervisado | [PDF](<teoria/Preparación de Datos para Aprendizaje no Supervisado.pdf>) | [MD](<teoria/Preparación de Datos para Aprendizaje no Supervisado.md>) |
| 7 | Evaluación de Modelos No Supervisados | [PDF](<teoria/Evaluación de Modelos No Supervisados.pdf>) | [MD](<teoria/Evaluación de Modelos No Supervisados.md>) |
| 9 | Fundamentos de Clustering | [PDF](<teoria/Fundamentos de Clustering.pdf>) | [MD](<teoria/Fundamentos de Clustering.md>) |
| 10 | Clustering Jerárquico | [PDF](<teoria/Clustering Jerárquico.pdf>) | [MD](<teoria/Clustering Jerárquico.md>) |
| 11 | Algoritmo de K-means | [PDF](<teoria/Algoritmo de K-means.pdf>) | [MD](<teoria/Algoritmo de K-means.md>) |
| 12 | Evaluación de Clusters | [PDF](<teoria/Evaluación de Clusters.pdf>) | [MD](<teoria/Evaluación de Clusters.md>) |
| 13 | DBSCAN y Algoritmos de Clustering Basados en Densidad | [PDF](<teoria/DBSCAN y Algoritmos de Clustering Basados en Densidad.pdf>) | [MD](<teoria/DBSCAN y Algoritmos de Clustering Basados en Densidad.md>) |
| 14 | Aplicaciones Prácticas del Clustering | [PDF](<teoria/Aplicaciones Prácticas del Clustering.pdf>) | [MD](<teoria/Aplicaciones Prácticas del Clustering.md>) |
| 16 | Comprender la Maldición de la Dimensión | [PDF](<teoria/Comprender la Maldición de la Dimensión.pdf>) | [MD](<teoria/Comprender la Maldición de la Dimensión.md>) |
| 17 | Estrategias de Selección de Variables | [PDF](<teoria/Estrategias de Selección de Variables.pdf>) | [MD](<teoria/Estrategias de Selección de Variables.md>) |
| 18 | Criterios de Selección de Modelos | [PDF](<teoria/Criterios de Selección de Modelos.pdf>) | [MD](<teoria/Criterios de Selección de Modelos.md>) |
| 19 | Selección de Variables con Regularización | [PDF](<teoria/Selección de Variables con Regularización.pdf>) | [MD](<teoria/Selección de Variables con Regularización.md>) |
| 22 | Principios de Reducción de la Dimensionalidad | [PDF](<teoria/Principios de Reducción de la Dimensionalidad.pdf>) | [MD](<teoria/Principios de Reducción de la Dimensionalidad.md>) |
| 23 | Análisis de Componentes Principales (PCA) | [PDF](<teoria/Análisis de Componentes Principales (PCA).pdf>) | [MD](<teoria/Análisis de Componentes Principales (PCA).md>) |
| 24 | Aplicaciones de PCA | [PDF](<teoria/Aplicaciones de PCA.pdf>) | [MD](<teoria/Aplicaciones de PCA.md>) |
| 25 | Análisis Discriminante Lineal (LDA) | [PDF](<teoria/Análisis Discriminante Lineal (LDA).pdf>) | [MD](<teoria/Análisis Discriminante Lineal (LDA).md>) |

Material del curso: [Programa del Curso](<teoria/Programa del Curso.pdf>) · [Cuestionario de Autoevaluación](<teoria/Cuestionario de Autoevaluación.pdf>).

### `notebooks/`

| Notebook | Tema |
|----------|------|
| [clustering_jerarquico_practica.ipynb](notebooks/clustering_jerarquico_practica.ipynb) | Práctica de la Clase 10: clustering jerárquico aglomerativo con `scipy` (`linkage`, `dendrogram`, `fcluster`); dendrogramas variando el número de clusters y los métodos de enlace (simple, completo, promedio, centroide, Ward) con distancia euclidiana. |
| [kmeans_practica.ipynb](notebooks/kmeans_practica.ipynb) | Práctica de la Clase 11: `KMeans` sobre `make_blobs`; visualización de clusters y centroides, análisis de silueta. |
| [evaluacion_clusters_practica.ipynb](notebooks/evaluacion_clusters_practica.ipynb) | Práctica de la Clase 12: elección del nº óptimo de clusters con el método del codo (inercia) y silueta. |
| [dbscan_practica.ipynb](notebooks/dbscan_practica.ipynb) | Práctica de la Clase 13: `DBSCAN`; ajuste de `eps` y `min_samples`, comparación con K-means. |
| [clustering_practica_complementaria.ipynb](notebooks/clustering_practica_complementaria.ipynb) | Práctica complementaria (Clase 14): K-means con `KFold` y `GridSearchCV`, imputación y escalado sobre el dataset `wine`. |
| [seleccion_variables_embedded_practica.ipynb](notebooks/seleccion_variables_embedded_practica.ipynb) | Práctica de la Clase 20 (métodos embedded): sobre el dataset `Iris`, análisis de dimensionalidad y correlación, K-means como línea base, elección del nº de clusters (codo vs. silueta) y selección de variables con `feature_importances` de `RandomForest`. |
| [aplicaciones_pca_practica.ipynb](notebooks/aplicaciones_pca_practica.ipynb) | Práctica de la Clase 24: `PCA` de sklearn sobre el dataset `wine`; estandarización con `StandardScaler`, varianza explicada por componente (`explained_variance_ratio_`), elección de `n_components` y clasificación comparando el rendimiento con y sin reducción. |

### `datasets/`
Datasets del módulo (se agregarán a medida que avance el módulo).

## Temario (plan de estudio)

Leyenda: ✅ material disponible · ⬜ pendiente.

### Módulo 1 — Bienvenida
- Clase 1 — Bienvenida: programa del curso, presentación, cuestionario de autoevaluación. ✅

### Módulo 2 — Introducción al Aprendizaje no Supervisado
- Clase 2 — Entorno de Desarrollo (IDE, instalaciones). ⬜
- Clase 3 — Principios del Aprendizaje no Supervisado. ✅
- Clase 4 — Aplicaciones del Aprendizaje no Supervisado. ✅
- Clase 5 — Técnicas Comunes en Aprendizaje no Supervisado. ✅
- Clase 6 — Preparación de Datos para Aprendizaje no Supervisado. ✅
- Clase 7 — Evaluación de Modelos No Supervisados. ✅
- Clase 8 — Checkpoint de contenidos. ⬜

### Módulo 3 — Clustering y K-means
- Clase 9 — Fundamentos de Clustering. ✅
- Clase 10 — Clustering Jerárquico (dendrogramas). ✅ + [práctica](notebooks/clustering_jerarquico_practica.ipynb)
- Clase 11 — Algoritmo de K-means. ✅ + [práctica](notebooks/kmeans_practica.ipynb)
- Clase 12 — Evaluación de Clusters (nº óptimo de clusters). ✅ + [práctica](notebooks/evaluacion_clusters_practica.ipynb)
- Clase 13 — DBSCAN y clustering basado en densidad. ✅ + [práctica](notebooks/dbscan_practica.ipynb)
- Clase 14 — Aplicaciones Prácticas del Clustering. ✅ + [práctica](notebooks/clustering_practica_complementaria.ipynb)
- Clase 15 — Checkpoint de contenidos. ⬜

### Módulo 4 — La Maldición de la Dimensión y Selección de Variables
- Clase 16 — Comprender la Maldición de la Dimensión. ✅
- Clase 17 — Estrategias de Selección de Variables (filtros, RFE). ✅
- Clase 18 — Criterios de Selección de Modelos (AIC, BIC). ✅
- Clase 19 — Selección de Variables con Regularización (Lasso, Ridge). ✅
- Clase 20 — Selección de Variables con Métodos Embedded (árboles, Random Forest). ✅ + [práctica](notebooks/seleccion_variables_embedded_practica.ipynb)
- Clase 21 — Checkpoint de contenidos. ⬜

### Módulo 5 — Reducción de la Dimensionalidad
- Clase 22 — Principios de Reducción de la Dimensionalidad. ✅
- Clase 23 — Análisis de Componentes Principales (PCA). ✅
- Clase 24 — Aplicaciones de PCA. ✅ + [práctica](notebooks/aplicaciones_pca_practica.ipynb)
- Clase 25 — Análisis Discriminante Lineal (LDA). ✅
- Clase 26 — T-SNE para Visualización. ⬜
- Clase 27 — UMAP para Reducción de Dimensionalidad. ⬜
- Clase 28 — Comparación de Técnicas de Reducción. ⬜
- Clase 29 — Checkpoint de contenidos. ⬜

### Módulo 6 — Cierre de Curso
- Clase 30 — Despedida (resumen del curso). ⬜
- Clase 31 — Evaluación Integral. ⬜

[← Volver al índice](../README.md)
