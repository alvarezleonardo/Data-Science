# Capa convolucional — Parte 2

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 04). El PDF original está en [`material/`](material/).

## 1. Padding

El **padding** consiste en añadir píxeles adicionales (normalmente ceros) alrededor de los bordes de la imagen de entrada, antes de aplicar la convolución.

Sin padding, aplicar un filtro reduce el tamaño de la imagen en cada convolución (como se vio en la Parte 1, donde una imagen 4×4 con un filtro 3×3 dio una salida 2×2). El padding permite que los filtros se apliquen correctamente también a los píxeles de los bordes, que de otro modo participan en menos convoluciones que los píxeles centrales.

La slide nombra tres variantes:

| Tipo | Qué hace |
|---|---|
| **Valid padding** | no agrega padding; la salida siempre es más chica que la entrada |
| **Same padding** | agrega el padding justo para que la salida tenga el **mismo** tamaño que la entrada |
| **Full padding** | agrega padding suficiente para que el filtro recorra la imagen desde que toca su primer píxel, dando una salida **mayor** que la entrada |

### Ejemplo desarrollado: same padding

La slide retoma la misma imagen 4×4 y el mismo filtro 3×3 de la Parte 1, pero ahora le agrega un borde de ceros (padding = 1), quedando una imagen de 6×6:

```
0  0  0  0  0  0
0  35 40 41 45 0
0  40 40 46 51 0
0  43 47 49 54 0
0  48 51 55 61 0
0  0  0  0  0  0
```

Con el mismo filtro `[1, 0, -1]` en cada fila, el filtro ahora sí puede recorrer 4 posiciones por fila y 4 por columna, y el mapa de características resultante es de **4×4** — el mismo tamaño que la imagen original:

| -80 | -12 | -16 | 87 |
|---|---|---|---|
| -127 | -18 | -23 | 136 |
| -138 | -19 | -28 | 150 |
| -98 | -13 | -17 | 104 |

Nótese que el bloque central de este resultado (`-18, -23, -19, -28`) coincide exactamente con el mapa de características de 2×2 obtenido en la Parte 1 sin padding: el padding no cambia lo que ya se calculaba, solo agrega las posiciones extra alrededor.

### La fórmula del tamaño de salida

El tamaño de salida de una convolución se calcula como:

```
tamaño_salida = (W − K + 2P) / S + 1
```

donde `W` es el tamaño de la entrada, `K` el tamaño del kernel, `P` el padding agregado por lado y `S` el stride.

Verificando el ejemplo de arriba: `W=4, K=3, P=1, S=1` → `(4 − 3 + 2×1)/1 + 1 = 3/1 + 1 = 4`. Coincide con la salida de 4×4.

Esto es exactamente lo que hacen las convoluciones del notebook del curso: `Conv2d(kernel_size=3, padding=1)` con stride 1 aplicado sobre una entrada de 32×32 da `(32 − 3 + 2)/1 + 1 = 32`, es decir, **conserva el tamaño espacial**. Por eso en la arquitectura de `CNN_pytorch.ipynb` el tamaño va 32 → 16 → 8 → 4: cada convolución con padding 1 y kernel 3 no reduce nada, y es el `MaxPool2d(2, 2)` posterior el que divide el tamaño a la mitad en cada bloque.

## 2. Stride

El **stride** define el número de píxeles que el filtro se desplaza en cada paso mientras recorre la imagen de entrada.

Las slides lo ilustran con un filtro de 2×2 que se desplaza sobre una entrada más grande: primero con stride 1 (el filtro se superpone bastante con la posición anterior) y después con stride 2 (el filtro salta directamente a la siguiente región, sin superposición, marcado en el diagrama con un recuadro punteado que separa cada aplicación del filtro de la siguiente).

- Un **stride mayor** da una salida más pequeña, porque el filtro se mueve más píxeles por vez y por lo tanto se aplican menos convoluciones en toda la imagen.
- Un **stride menor** implica más superposición entre aplicaciones sucesivas del filtro, lo que puede llevar a mayor resolución en el mapa de características y a capturar más detalle.

El stride es el parámetro `S` de la fórmula de la sección anterior: a mayor `S`, menor el tamaño de salida.

## 3. Función de activación: ReLU

Después de calcular el mapa de características, se le aplica una función de activación no lineal. La que usa la capa convolucional es la **unidad lineal rectificada (ReLU)**:

```
f(x) = 0   si x < 0
f(x) = x   si x ≥ 0
```

Es decir: deja pasar sin cambios los valores positivos y convierte en cero cualquier valor negativo.

La slide lo aplica sobre el resultado de la convolución con padding de la sección 1. Tomando la primera fila del mapa de características (`-80, -12, -16, 87`), después de ReLU queda:

| 0 | 0 | 0 | 87 |
|---|---|---|---|

Y del mismo modo con las filas restantes (`-127, -18, -23, 136` → `0, 0, 0, 136`; etc.), quedando finalmente el **mapa de características** con todos los valores negativos anulados y solo los positivos conservados.

> **Nota:** ReLU tiene el mismo rol acá que en cualquier red densa: sin una no linealidad entre capas, apilar convoluciones sería matemáticamente equivalente a una sola convolución más grande (la composición de transformaciones lineales es lineal). ReLU es la que le permite a la red aproximar funciones no lineales apilando capas convolucionales.

## 4. Relación con el resto del módulo

- Este bloque completa la mecánica de la capa convolucional que arrancó en [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md): ahí se vio el producto punto filtro-imagen: acá, cómo controlar el tamaño de la salida (padding, stride) y cómo se introduce la no linealidad (ReLU).
- La fórmula `(W − K + 2P)/S + 1` es la misma que explica por qué, en el notebook `CNN_pytorch.ipynb`, las tres convoluciones (`Conv2d(3→32→64→128, kernel 3, padding 1)`) no cambian el tamaño espacial, y es el `MaxPool2d` el responsable de la reducción 32→16→8→4.
- ReLU es la activación que efectivamente usan las capas `Conv2d` del notebook entre convolución y pooling; el mapa de características "con ceros" de este bloque es el mismo tipo de tensor que produce cada capa convolucional de la red antes de pasar al pooling.
- Falta antes de las capas totalmente conectadas la capa de **agrupamiento** (pooling): no está entre los PDFs de esta tanda, pero es la pieza que el diagrama de [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](01%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%201.md) ubica entre la convolución y el aplanamiento.
