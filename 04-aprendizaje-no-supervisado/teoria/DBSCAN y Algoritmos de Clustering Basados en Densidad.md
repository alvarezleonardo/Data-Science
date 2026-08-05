# DBSCAN y Algoritmos de Clustering Basados en Densidad

> Conversión a Markdown de la slide del curso (Clase 13). El PDF original está al lado.

## ¿Qué es DBSCAN?

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo de clustering **basado en densidad**: identifica clusters como **regiones de alta densidad** separadas por **regiones de baja densidad**.

A diferencia de los métodos basados en centroides (K-means) o jerárquicos, DBSCAN:
- **No requiere** especificar el número de clusters de antemano.
- Puede identificar clusters de **forma arbitraria** (no solo esféricos).
- Detecta **ruido / outliers**.

## 1. Funcionamiento del algoritmo

### Parámetros principales
- **Epsilon (`ε`):** distancia máxima para considerar que dos puntos están en la misma **vecindad**.
- **MinPts (número mínimo de puntos):** cantidad mínima de puntos para formar un cluster (incluyendo el punto central).

### Clasificación de los puntos
- **Punto Central (core):** tiene al menos **MinPts** puntos dentro de una distancia `ε` (se incluye a sí mismo).
- **Punto de Borde (border):** no es central, pero está dentro de `ε` de un punto central.
- **Punto de Ruido (noise):** no es central ni de borde → **outlier**.

### Expansión del cluster
- Comienza con un **punto central** y expande el cluster a los **puntos de borde** dentro de `ε`.
- Verifica si esos puntos de borde también pueden ser **nuevos puntos centrales**; el proceso se repite de forma **recursiva**.

### Terminación
- Continúa hasta que todos los puntos estén **asignados a un cluster** o clasificados como **ruido**.

## 2. Comparación con K-means y clustering jerárquico

En datos con **formas no esféricas** (anillos concéntricos, medias lunas, espirales), DBSCAN separa correctamente los grupos siguiendo la densidad, mientras que **K-means** —que parte de centroides y fronteras aproximadamente esféricas— los corta mal.

| Método | Nº clusters | Formas | Ruido | Escala |
|--------|:-----------:|--------|:-----:|--------|
| **K-means** | Hay que fijarlo | Esféricas | Sensible | Rápido, escala bien |
| **DBSCAN** | Automático | **Arbitrarias** | **Detecta ruido** | Sensible a `ε`/MinPts |
| **Jerárquico** | Se decide al cortar | Según enlace | Según método | Menos eficiente en datos grandes |

**Cuándo usar cada uno:**
- **K-means:** cuando tenés una idea clara del número de clusters y son aproximadamente **esféricos**. Menos adecuado con ruido o formas arbitrarias.
- **DBSCAN:** ideal para clusters de **formas arbitrarias** y para **manejar ruido**. No requiere fijar `k`, pero es sensible a la elección de `ε` y **MinPts**.
- **Clustering jerárquico:** da una visión detallada de la estructura (dendrograma) y no requiere fijar `k`, pero es **menos eficiente** en grandes volúmenes.

> Práctica asociada: [notebooks/dbscan_practica.ipynb](../notebooks/dbscan_practica.ipynb) — `DBSCAN` de sklearn, ajuste de `eps` y `min_samples`, y comparación con K-means.
