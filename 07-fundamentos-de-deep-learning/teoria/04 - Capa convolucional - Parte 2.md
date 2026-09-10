# Capa convolucional — Parte 2

> Conversión a Markdown de las slides del curso (Clase 9 — Capa convolucional, bloque 04). El PDF original está al lado. Continúa la [Parte 1](<03 - Capa convolucional - Parte 1.md>), que definió la operación y el ejemplo numérico.

## 1. Padding

El **padding** consiste en **añadir píxeles adicionales alrededor de los bordes** de la imagen de entrada antes de aplicar la convolución.

Los dos motivos que da el material:

- **Sin padding, cada convolución achica la imagen.** En el ejemplo de la Parte 1, 4×4 con filtro 3×3 dio 2×2. Apilar diez capas así deja la imagen en nada; el padding permite hacer redes profundas.
- **El padding permite que los filtros se apliquen correctamente a los píxeles de los bordes.** Sin él, un píxel de la esquina participa de una sola ventana mientras que uno del centro participa de nueve: la información del borde queda subrepresentada.

### El ejemplo

A la imagen de 4×4 se le agrega **un anillo de ceros**, y queda de 6×6:

```
 0   0   0   0   0   0
 0  35  40  41  45   0
 0  40  40  46  51   0
 0  43  47  49  54   0
 0  48  51  55  61   0
 0   0   0   0   0   0
```

Con el mismo filtro de bordes verticales (`1 0 -1` por fila), el recorrido da un mapa de **4×4** — el mismo tamaño que la entrada original:

```
 -80   -12   -16    87
-127   -18   -23   136
-138   -19   -28   150
 -98   -13   -17   104
```

Verificación de la primera posición, donde el filtro se apoya mayormente sobre el relleno:

```
0×1 + 0×0 + 0×(-1) + 0×1 + 35×0 + 40×(-1) + 0×1 + 40×0 + 40×(-1) = -80
```

Dos cosas para mirar en ese resultado:

- **El bloque central `-18 -23 / -19 -28` es idéntico al de la Parte 1.** El padding no cambia lo que ya se calculaba: solo agrega las posiciones del borde que antes no existían.
- **Los valores de los bordes son enormes** (-138, 150) comparados con los del centro (-19, -28). Es un artefacto del relleno: el filtro compara píxeles reales contra ceros, y esa diferencia artificial se ve como un borde muy marcado. Es el precio del padding con ceros, y la razón por la que existen alternativas como el relleno por reflejo o por réplica.

### Los tres tipos

El material nombra **Valid Padding, Same Padding y Full Padding**:

| Tipo | Qué hace | Tamaño de salida (entrada `N`, filtro `F`, stride 1) |
|---|---|---|
| **Valid** | sin relleno; solo ventanas que entran completas | `N - F + 1` — más chica |
| **Same** | relleno tal que la salida conserve el tamaño | `N` — igual |
| **Full** | relleno máximo; cada píxel participa de todas las ventanas posibles | `N + F - 1` — más grande |

El ejemplo de arriba es **same padding**: con `F = 3` y `P = 1`, la fórmula `(N + 2P - F)/S + 1` da `(4 + 2 - 3)/1 + 1 = 4`.

En código, `padding='same'` en Keras y `padding=1` en `nn.Conv2d` de PyTorch son la misma decisión — y es exactamente lo que usan los notebooks de las clases 12 y 13, para que el tamaño lo controle el pooling y no la convolución.

## 2. Stride

El **stride** define **cuántos píxeles se desplaza el filtro en cada paso** mientras recorre la imagen. Hasta acá el ejemplo usó stride 1: un píxel por vez.

| Stride | Efecto |
|---|---|
| **mayor** | salida más chica: el filtro se mueve más píxeles por vez, así que se aplican menos convoluciones sobre la imagen |
| **menor** | más superposición entre aplicaciones sucesivas del filtro, lo que da mayor resolución y captura más detalle |

Es un compromiso directo entre **detalle** y **costo**. Con stride 2 la salida queda aproximadamente a la mitad en cada dimensión, o sea a un cuarto de valores: la red corre mucho más rápido pero pierde precisión espacial.

En la fórmula, el stride es el divisor:

```
salida = (N + 2P - F) / S + 1
```

Con `N = 32`, `F = 3`, `P = 1`: stride 1 da 32; stride 2 da 16.

Hay dos maneras de reducir el tamaño espacial —stride mayor que 1, o una capa de *pooling*—. Los notebooks del módulo eligen la segunda: convolución con stride 1 y `padding='same'` para no perder tamaño, y después `MaxPool2D(2,2)` para reducir a la mitad. Así queda separado "extraer características" de "reducir resolución".

## 3. Función de activación

Al mapa de características se le aplica una función de activación, elemento por elemento. La que usa el material es **ReLU** (*Rectified Linear Unit*, unidad lineal rectificada):

```
ReLU(x) = max(0, x)
```

Deja pasar los valores positivos tal cual y lleva a cero los negativos.

Aplicada al mapa con padding de la sección 1:

```
   0     0     0    87
   0     0     0   136
   0     0     0   150
   0     0     0   104
```

Ese es el **mapa de características** que sale de la capa.

Por qué se hace:

- **Sin activación no habría profundidad real.** La convolución es una operación lineal; encadenar convoluciones sin nada en el medio equivale a una sola convolución. La no linealidad es lo que permite que las capas construyan representaciones cada vez más complejas.
- **ReLU es barata y no satura.** Es un `max`, y su derivada es 0 o 1: no sufre el desvanecimiento del gradiente que tienen sigmoide y tanh en redes profundas (cap. 33 del manual).

Y qué se pierde en este ejemplo: el filtro daba negativo para los bordes "de claro a oscuro" y positivo para los inversos. ReLU borra toda una mitad de esa información. En una red real no es un problema, porque hay decenas de filtros y el entrenamiento aprende el filtro de signo opuesto cuando esa dirección importa.

## 4. Cierre de la capa convolucional

El bloque completo de una capa convolucional, entonces:

```
entrada → convolución (filtros + padding + stride) → activación ReLU → mapa de características
```

Ese mapa es lo que recibe la **capa de agrupamiento** (*pooling*) de la Clase 10, que lo reduce antes de pasarlo a la capa convolucional siguiente.
