# Capa convolucional — Parte 1

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 03). El PDF original está en [`material/`](material/).

## 1. Qué hace la capa convolucional

Su principal función es **extraer características de la entrada** mediante la aplicación de **filtros (kernels)** que detectan patrones.

Estos filtros se deslizan (**convolucionan**) sobre la imagen de entrada. Cada filtro es una pequeña matriz de pesos que se ajusta durante el entrenamiento. El filtro se aplica a un área de la imagen y se calcula un producto entre los píxeles de entrada y el filtro; luego el filtro se desplaza un poco y repite el proceso hasta que recorrió toda la imagen.

Esa operación —producto elemento a elemento entre filtro y región de la imagen, sumado en un único valor— es exactamente la **convolución** que le da nombre a la red.

## 2. El ejemplo numérico paso a paso

La slide desarrolla un ejemplo completo con una imagen de 4×4 (un solo canal) y un filtro de 3×3:

**Imagen de entrada (4×4):**

| 35 | 40 | 41 | 45 |
|---|---|---|---|
| 40 | 40 | 46 | 51 |
| 43 | 47 | 49 | 54 |
| 48 | 51 | 55 | 61 |

**Filtro (kernel, 3×3):**

| 1 | 0 | -1 |
|---|---|---|
| 1 | 0 | -1 |
| 1 | 0 | -1 |

Este filtro en particular es un **detector de bordes verticales**: resta la columna derecha de la región a la izquierda, así que da un valor alto (en magnitud) donde hay un cambio brusco de intensidad de izquierda a derecha, y cercano a cero donde la región es uniforme.

El filtro recorre la imagen de a un píxel por vez (stride 1), sin agregar padding, y en cada posición calcula el producto punto entre los 9 valores del filtro y los 9 píxeles que tiene debajo:

| Posición del filtro | Cálculo | Resultado |
|---|---|---|
| Esquina superior izquierda | 35×1 + 40×0 + 41×(-1) + 40×1 + 40×0 + 46×(-1) + 43×1 + 47×0 + 49×(-1) = 35 − 41 + 40 − 46 + 43 − 49 | **-18** |
| Un paso a la derecha | 40×1 + 41×0 + 45×(-1) + 40×1 + 46×0 + 51×(-1) + 47×1 + 49×0 + 54×(-1) = 40 − 45 + 40 − 51 + 47 − 54 | **-23** |
| Una fila abajo, columna inicial | 40×1 + 40×0 + 46×(-1) + 43×1 + 47×0 + 49×(-1) + 48×1 + 51×0 + 55×(-1) = 40 − 46 + 43 − 49 + 48 − 55 | **-19** |
| Una fila abajo, un paso a la derecha | 40×1 + 46×0 + 51×(-1) + 47×1 + 49×0 + 54×(-1) + 51×1 + 55×0 + 61×(-1) = 40 − 51 + 47 − 54 + 51 − 61 | **-28** |

El resultado final, el **mapa de características** (*feature map*), es una matriz de 2×2:

| -18 | -23 |
|---|---|
| -19 | -28 |

La imagen de entrada era 4×4 y el filtro 3×3; el resultado dio 2×2. Esa reducción de tamaño es la consecuencia directa de no usar padding, y se retoma en la Parte 2 con la fórmula general.

## 3. De un canal a varios: filtros en volumen

Las últimas slides extienden el ejemplo a una imagen de **4×4×3** (tres canales, por ejemplo RGB) con un filtro de **3×3×3**: el filtro deja de ser una matriz plana y pasa a ser un volumen con la misma profundidad que la imagen de entrada.

La convolución sobre varios canales funciona igual en la idea, pero suma tres productos punto en vez de uno: el filtro se aplica en simultáneo sobre los tres canales (rojo, verde, azul) y los tres resultados parciales se suman en un único número por posición, de la misma forma en que antes se sumaban los 9 productos de un filtro 3×3.

> **Nota:** las slides que muestran el filtro 3×3×3 (build de esta sección) reutilizan como resultado la misma matriz `-18 -23 / -19 -28` calculada para el caso de un solo canal, en vez de recalcular la suma sobre los tres canales. Es una simplificación visual del deck para no introducir otra tabla de números — sirve para mostrar la idea de que la salida de un filtro sobre un volumen sigue siendo un mapa 2D (un canal), no para tomar esos números como el resultado real de una convolución de 3 canales.

Esta es la razón por la que, sin importar cuántos canales tenga la entrada, **un filtro produce un único mapa de características de 2 dimensiones**: la profundidad del filtro absorbe la profundidad de la entrada. Es lo mismo que ocurre en la primera capa convolucional del notebook del curso: `Conv2d(3, 32, kernel_size=3, padding=1)` toma una entrada de 3 canales y aplica 32 filtros distintos, cada uno de forma 3×3×3, para producir 32 mapas de características (uno por filtro).

## 4. Cuántos parámetros tiene un filtro

Un filtro no es solo la matriz de pesos: además de los pesos, cada filtro tiene **un sesgo (bias)**. La cantidad de parámetros de un filtro es:

```
parámetros por filtro = (kernel × kernel × canales_de_entrada) + 1 sesgo
```

Verificado contra la primera capa convolucional de `CNN_pytorch.ipynb` (`Conv2d(3, 32, kernel_size=3, padding=1)`):

- Un filtro: 3×3×3 + 1 = 27 + 1 = **28 parámetros**.
- 32 filtros: 28 × 32 = **896 parámetros**, que coincide exactamente con lo que reporta el `summary()` del modelo para esa capa.

## 5. Relación con el resto del módulo

- Este bloque introduce la operación de convolución en sí; la Parte 2 (ver [`Capa convolucional - P2.md`](Capa%20convolucional%20-%20P2.md)) agrega **padding**, **stride** y la función de activación **ReLU** que se aplica sobre el mapa de características que se calculó acá.
- La representación de la imagen en canales, vista en [`02 - Introducción a las redes neuronales convolucionales - Parte 2.md`](02%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%202.md), es la que se generaliza acá al filtro en volumen (3×3×3).
- Los tres bloques convolucionales del notebook (`Conv2d(3→32→64→128, kernel 3, padding 1)`) son exactamente esta operación aplicada tres veces, cada vez con más filtros y más canales de entrada.
