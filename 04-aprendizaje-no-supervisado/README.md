# 04 — Aprendizaje no supervisado

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 6 | 31 | ✅ Aprobado (6/6) |

Técnicas de aprendizaje no supervisado: clustering (K-Means, jerárquico, DBSCAN), reducción de dimensionalidad (PCA, LDA, t-SNE, UMAP) y selección de variables. Curso "ML3" del programa de Digital House.

> **Apuntes + referencia técnica** del módulo (qué es cada técnica, cuándo aplicarla, hiperparámetros, cómo se evalúa y snippets de `scikit-learn`): ver el documento maestro **[manual/ → Parte VI](../manual/06-aprendizaje-no-supervisado.md#parte-vi--aprendizaje-no-supervisado)** (caps. 15-23). Pensado para repaso de examen.

## Contenido

> Los **24 PDF originales** del curso están en [`teoria/material/`](teoria/material/); en `teoria/` quedan sus conversiones a Markdown.

### `teoria/`
Slides del curso (PDF + conversión `.md`):

| Clase | Tema | PDF | Markdown |
|:-----:|------|-----|----------|
| 3 | Principios del Aprendizaje no Supervisado | [PDF](<teoria/material/principios-del-aprendizaje-no-supervisado.pdf>) | [MD](<teoria/principios-del-aprendizaje-no-supervisado.md>) |
| 4 | Aplicaciones del Aprendizaje no Supervisado | [PDF](<teoria/material/aplicaciones-del-aprendizaje-no-supervisado.pdf>) | [MD](<teoria/aplicaciones-del-aprendizaje-no-supervisado.md>) |
| 5 | Técnicas Comunes en Aprendizaje no Supervisado | [PDF](<teoria/material/tecnicas-comunes-en-aprendizaje-no-supervisado.pdf>) | [MD](<teoria/tecnicas-comunes-en-aprendizaje-no-supervisado.md>) |
| 6 | Preparación de Datos para Aprendizaje no Supervisado | [PDF](<teoria/material/preparacion-de-datos-para-aprendizaje-no-supervisado.pdf>) | [MD](<teoria/preparacion-de-datos-para-aprendizaje-no-supervisado.md>) |
| 7 | Evaluación de Modelos No Supervisados | [PDF](<teoria/material/evaluacion-de-modelos-no-supervisados.pdf>) | [MD](<teoria/evaluacion-de-modelos-no-supervisados.md>) |
| 9 | Fundamentos de Clustering | [PDF](<teoria/material/fundamentos-de-clustering.pdf>) | [MD](<teoria/fundamentos-de-clustering.md>) |
| 10 | Clustering Jerárquico | [PDF](<teoria/material/clustering-jerarquico.pdf>) | [MD](<teoria/clustering-jerarquico.md>) |
| 11 | Algoritmo de K-means | [PDF](<teoria/material/algoritmo-de-k-means.pdf>) | [MD](<teoria/algoritmo-de-k-means.md>) |
| 12 | Evaluación de Clusters | [PDF](<teoria/material/evaluacion-de-clusters.pdf>) | [MD](<teoria/evaluacion-de-clusters.md>) |
| 13 | DBSCAN y Algoritmos de Clustering Basados en Densidad | [PDF](<teoria/material/dbscan-y-algoritmos-de-clustering-basados-en-densidad.pdf>) | [MD](<teoria/dbscan-y-algoritmos-de-clustering-basados-en-densidad.md>) |
| 14 | Aplicaciones Prácticas del Clustering | [PDF](<teoria/material/aplicaciones-practicas-del-clustering.pdf>) | [MD](<teoria/aplicaciones-practicas-del-clustering.md>) |
| 16 | Comprender la Maldición de la Dimensión | [PDF](<teoria/material/comprender-la-maldicion-de-la-dimension.pdf>) | [MD](<teoria/comprender-la-maldicion-de-la-dimension.md>) |
| 17 | Estrategias de Selección de Variables | [PDF](<teoria/material/estrategias-de-seleccion-de-variables.pdf>) | [MD](<teoria/estrategias-de-seleccion-de-variables.md>) |
| 18 | Criterios de Selección de Modelos | [PDF](<teoria/material/criterios-de-seleccion-de-modelos.pdf>) | [MD](<teoria/criterios-de-seleccion-de-modelos.md>) |
| 19 | Selección de Variables con Regularización | [PDF](<teoria/material/seleccion-de-variables-con-regularizacion.pdf>) | [MD](<teoria/seleccion-de-variables-con-regularizacion.md>) |
| 22 | Principios de Reducción de la Dimensionalidad | [PDF](<teoria/material/principios-de-reduccion-de-la-dimensionalidad.pdf>) | [MD](<teoria/principios-de-reduccion-de-la-dimensionalidad.md>) |
| 23 | Análisis de Componentes Principales (PCA) | [PDF](<teoria/material/analisis-de-componentes-principales-pca.pdf>) | [MD](<teoria/analisis-de-componentes-principales-pca.md>) |
| 24 | Aplicaciones de PCA | [PDF](<teoria/material/aplicaciones-de-pca.pdf>) | [MD](<teoria/aplicaciones-de-pca.md>) |
| 25 | Análisis Discriminante Lineal (LDA) | [PDF](<teoria/material/analisis-discriminante-lineal-lda.pdf>) | [MD](<teoria/analisis-discriminante-lineal-lda.md>) |
| 26 | T-SNE para Visualización | [PDF](<teoria/material/t-sne-para-visualizacion.pdf>) | [MD](<teoria/t-sne-para-visualizacion.md>) |
| 27 | UMAP para Reducción de Dimensionalidad | [PDF](<teoria/material/umap-para-reduccion-de-dimensionalidad.pdf>) | [MD](<teoria/umap-para-reduccion-de-dimensionalidad.md>) |
| 28 | Comparación de Técnicas de Reducción de Dimensionalidad | [PDF](<teoria/material/comparacion-de-tecnicas-de-reduccion-de-dimensionalidad.pdf>) | [MD](<teoria/comparacion-de-tecnicas-de-reduccion-de-dimensionalidad.md>) |

Material del curso: [Programa del Curso](<teoria/material/programa-del-curso.pdf>) · [Cuestionario de Autoevaluación](<teoria/material/cuestionario-de-autoevaluacion.pdf>).

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
| [lda_practica.ipynb](notebooks/lda_practica.ipynb) | Práctica de la Clase 25: `LinearDiscriminantAnalysis` de sklearn; proyección supervisada que maximiza la separación entre clases y comparación con la proyección de PCA. |
| [tsne_practica.ipynb](notebooks/tsne_practica.ipynb) | Práctica de la Clase 26: `TSNE` de sklearn (`sklearn.manifold`); visualización en 2D de datos de alta dimensión, efecto de `perplexity` y exploración de clusters. |
| [umap_practica.ipynb](notebooks/umap_practica.ipynb) | Práctica de la Clase 27: `UMAP` (paquete `umap-learn`); proyección a 2D preservando estructura local y global, ajuste de `n_neighbors` y `min_dist`. Requiere `pip install umap-learn`. |

### `datasets/`
Datasets del módulo (se agregarán a medida que avance el módulo).

## Temario (plan de estudio)

Leyenda: ✅ material disponible · ⬜ pendiente.

> **Cierre de contenido:** todas las clases con material teórico/práctico del curso (23) están procesadas y consolidadas. Las clases restantes son de **entorno** (C2), **checkpoints** de contenidos (C8, C15, C21, C29) y **cierre** (C30 despedida/resumen, C31 evaluación integral): no tienen slides propias, por eso figuran como ⬜. El módulo está ✅ Aprobado: la evaluación integral ya fue rendida.

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
- Clase 25 — Análisis Discriminante Lineal (LDA). ✅ + [práctica](notebooks/lda_practica.ipynb)
- Clase 26 — T-SNE para Visualización. ✅ + [práctica](notebooks/tsne_practica.ipynb)
- Clase 27 — UMAP para Reducción de Dimensionalidad. ✅ + [práctica](notebooks/umap_practica.ipynb)
- Clase 28 — Comparación de Técnicas de Reducción. ✅
- Clase 29 — Checkpoint de contenidos. ⬜

### Módulo 6 — Cierre de Curso
- Clase 30 — Despedida (resumen del curso). ⬜
- Clase 31 — Evaluación Integral. ⬜

[← Volver al índice](../README.md)
