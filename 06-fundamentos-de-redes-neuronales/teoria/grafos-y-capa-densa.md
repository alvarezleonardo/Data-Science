# Grafos y capa densa

> Conversión a Markdown de la slide del curso (unidad "Funciones de activación y estructura de la red", bloque 02). El PDF original está en [`material/`](material/).

## 1. La red como grafo

Un **grafo** es una representación que describe cómo las unidades (neuronas) están conectadas entre sí. Sus dos elementos:

- **Nodos** (vértices): las neuronas.
- **Aristas** (enlaces): las conexiones entre neuronas. Cada arista lleva un **peso**, que es lo que la red ajusta al entrenar.

## 2. Capa densa

En una **capa densa**, también llamada **capa totalmente conectada** (*fully connected*), **cada neurona de la capa está conectada a todas las neuronas de la capa anterior**.

Consecuencia práctica: si la capa anterior tiene `n` neuronas y la capa densa tiene `m`, hay `n × m` pesos más `m` sesgos. Es lo que hace que el número de parámetros crezca rápido al agrandar la red (§34).

## 3. El diagrama de la slide

La slide muestra una red de cuatro bloques, de izquierda a derecha:

- **Capa de entrada:** nodos `x₁, x₂, x₃, …, xₙ`. Uno por característica del dataset; no calculan nada, solo reciben los datos.
- **Capas ocultas:** dos capas de nodos rotulados `Σf`. El rótulo es literal: cada neurona hace la **suma ponderada** (`Σ`) y le aplica la **función de activación** (`f`).
- **Capa de salida:** nodos `y₁, y₂`. Uno por clase en clasificación multiclase (o uno solo en regresión / clasificación binaria).

Todas las aristas de una capa a la siguiente están dibujadas: es exactamente la definición de capa densa.

## 4. Relación con el resto del módulo

- El nodo `Σf` es el **perceptrón** (§27): suma ponderada más activación. La red no es otra cosa que muchos de ellos conectados.
- La `f` de cada nodo es la función de activación (§32); si fuera lineal, toda la red colapsaría a un perceptrón simple.
- En `scikit-learn`, `hidden_layer_sizes=(100, 150)` describe exactamente este grafo: dos capas ocultas densas, de 100 y 150 nodos. Los pesos de cada capa quedan en `mlp.coefs_` y los sesgos en `mlp.intercepts_`.
