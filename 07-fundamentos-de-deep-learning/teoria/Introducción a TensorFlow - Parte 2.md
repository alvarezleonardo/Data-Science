# Introducción a TensorFlow - Parte 2

> Conversión a Markdown de las slides del curso (Clase 3 — Introducción a TensorFlow, bloque 2). El PDF original está al lado.

## 1. Acceso a los datos

Keras proporciona acceso a varios conjuntos de datos (*datasets*) comunes directamente a través de su API, lo que facilita su uso en la construcción de modelos de aprendizaje profundo sin necesidad de cargar y preprocesar datos manualmente.

```python
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

## 2. Definir un modelo

Keras ofrece dos formas de definir la arquitectura de un modelo:

- **Modelo Secuencial (`Sequential`)**
- **Modelo Funcional (`Functional API`)**

```python
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(8,)))
model.add(layers.Dense(3, activation='softmax'))
```

> **Nota:** la slide solo menciona las dos opciones y muestra código de `Sequential`, sin explicar la diferencia. En la práctica:
>
> | | `Sequential` | `Functional API` |
> |---|---|---|
> | **Estructura** | Una pila lineal de capas, una entrada y una salida. | Un grafo de capas: se conectan explícitamente pasándose como funciones (`x = layers.Dense(10)(input)`). |
> | **Entradas/salidas** | Una sola entrada, una sola salida. | Permite **múltiples entradas y/o salidas**. |
> | **Topología** | Solo conexiones en cadena, capa tras capa. | Permite conexiones no lineales: ramas, *skip connections*, capas compartidas. |
> | **Cuándo usarla** | Alcanza para la mayoría de los modelos simples de clasificación/regresión. | Necesaria para arquitecturas más complejas (por ejemplo, un modelo con dos entradas que se combinan en una capa intermedia). |

## 3. Compilar el modelo

Compilar el modelo es configurarlo con la información necesaria para su entrenamiento. Se especifican el **optimizador**, la **función de pérdida** y las **métricas** que se usarán.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 3.1. Optimizador

Los optimizadores son los algoritmos que ajustan los pesos del modelo para reducir la función de pérdida durante el entrenamiento (ver `Optimización.md` del módulo 06 para el mecanismo de fondo: descenso de gradiente, tasa de aprendizaje).

### 3.2. Función de pérdida (`loss`)

| Función | Cuándo usarla |
|---|---|
| `binary_crossentropy` | Problemas de clasificación **binaria** (dos clases). |
| `categorical_crossentropy` | Clasificación **multiclase** donde las etiquetas están en formato **one-hot**. |
| `sparse_categorical_crossentropy` | Clasificación multiclase donde las etiquetas **no** están en one-hot, sino que son **enteros** que representan las clases. |
| `mean_squared_error` (`mse`) | Problemas de **regresión**. |

> **Nota — la diferencia clave entre las dos crossentropy multiclase:** ambas resuelven el mismo problema (clasificación con más de dos clases) y dan el mismo resultado matemático; lo único que cambia es **cómo vienen codificadas las etiquetas**:
>
> - Con `categorical_crossentropy`, `y_train` debe estar en **one-hot**: para 3 clases, la clase 1 es `[0, 1, 0]`.
> - Con `sparse_categorical_crossentropy`, `y_train` son directamente **enteros**: la clase 1 es `1`.
>
> Usar `sparse_categorical_crossentropy` ahorra el paso de convertir las etiquetas con `to_categorical` (o `OneHotEncoder`), y es la opción más común cuando el dataset ya trae las etiquetas como enteros (como `cifar10`, que se carga arriba).

### 3.3. Métricas

Las métricas se usan para evaluar el rendimiento del modelo durante el entrenamiento y la validación, pero **no** intervienen en el ajuste de los pesos (a diferencia de la función de pérdida).

- `accuracy`: calcula la precisión del modelo.

## 4. Relación con el resto del módulo

- El código de esta parte (`cifar10`, `Sequential`, `compile`) es el paso intermedio entre los fundamentos conceptuales de la **Parte 1** y el entrenamiento/evaluación de la **Parte 3**.
- El optimizador (`adam`) y la función de pérdida retoman directamente lo visto en `Optimización.md` y `Funciones de Pérdida.md` del módulo 06, ahora en su forma de "un string en `compile`" en vez de una implementación manual.
