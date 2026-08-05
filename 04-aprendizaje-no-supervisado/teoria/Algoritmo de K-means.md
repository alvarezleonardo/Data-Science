# Algoritmo de K-means

> Conversión a Markdown de la slide del curso (Clase 11). El PDF original está al lado.

## 1. ¿Qué es el algoritmo k-means?

Es un método de segmentación; el **k-means** es, quizás, el instrumento **más clásico** para agrupar. Se fija de forma previa un **número `k` de grupos** y el algoritmo busca los **mejores centroides** para segmentar. Su objetivo: que los miembros de cada agrupación estén **lo más próximos posible a su centroide**.

## 2. Proceso

El algoritmo funciona de manera **iterativa**: actualiza el centro de los clústeres de modo que va **reduciendo las distancias** con cada uno de sus individuos.

Pasos:
1. **Determinar los centroides iniciales** (semillas) de los `k` grupos.
2. **Asignar** cada observación al centroide **más cercano** según la distancia (ej. euclídea).
3. Con las observaciones clasificadas, **recalcular** los centroides de los `k` grupos. Si el desplazamiento entre nuevos y viejos centroides supera un **criterio de convergencia**, se vuelve al paso 2.
4. El proceso **termina** al cumplir el criterio de convergencia. Además se fija un **número máximo de iteraciones** que no debe superarse.

## 3. Funcionamiento (paso a paso)

1. **Conjunto de datos:** se parte de los puntos sin etiquetar.
2. **Inicialización:** elegir el número de clústeres `K` y **seleccionar `K` puntos iniciales** (aleatoriamente o con un método más sofisticado, p. ej. k-means++).
3. **Asignación de clústeres:** para cada punto, se calcula la distancia a cada uno de los `K` centroides y se asigna al **más cercano** (distancia euclidiana u otra).
4. **Actualización de centroides:** se recalcula el centroide de cada clúster como el **promedio** de todos sus puntos.
5. **Repetición:** se repiten asignación y actualización hasta que los centroides **ya no cambien significativamente** o se alcance el máximo de iteraciones (convergencia).
6. **Resultados:** cada punto queda asignado a uno de los `K` clústeres y los centroides finales representan sus centros.

## 4. Criterios de parada

- **Los centroides dejan de cambiar:** tras varias iteraciones no se mueven → se asume convergencia.
- **Los puntos dejan de cambiar de clúster:** ya no hay intercambio de puntos entre clústeres → modelo entrenado.
- **Límite de iteraciones:** se fija un máximo; al alcanzarlo se detiene el entrenamiento (se asume que no mejorará drásticamente).

> Recordar: `k` es un **hiperparámetro** (se define antes de entrenar). Para elegirlo se usan el **método del codo** y el **índice de silueta** (ver Clase 12).

> Práctica asociada: [notebooks/kmeans_practica.ipynb](../notebooks/kmeans_practica.ipynb) — `KMeans` de sklearn sobre `make_blobs`, visualización de clusters y centroides, y análisis de silueta.
