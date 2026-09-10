# Capa convolucional — Parte 1

> Conversión a Markdown de las slides del curso (Clase 9 — Capa convolucional, bloque 03). El PDF original está al lado. La [Parte 2](<04 - Capa convolucional - Parte 2.md>) sigue con padding, stride y función de activación.

## 1. Qué hace la capa convolucional

Su función principal es **extraer características de la entrada** aplicando **filtros** (*kernels*) que detectan patrones.

El mecanismo, tal como lo describe la slide:

1. Los filtros se **deslizan** (convolucionan) sobre la imagen de entrada.
2. Cada filtro es una **matriz chica de pesos** que se ajusta durante el entrenamiento.
3. El filtro se aplica sobre un área de la imagen y se calcula un producto entre los píxeles y el filtro.
4. El filtro se desplaza un poco y repite el proceso hasta recorrer toda la imagen.

Los dos puntos que hacen toda la diferencia con una capa densa:

- **Los pesos del filtro se aprenden.** Nadie programa "detector de bordes verticales": el descenso por gradiente llega a esos valores solo, porque le sirven para bajar la pérdida.
- **El mismo filtro se usa en toda la imagen.** Es el *weight sharing*: un filtro de 3×3 sobre un canal son 9 pesos (más el sesgo), sin importar si la imagen es de 32×32 o de 1024×1024. Ahí está el ahorro frente al MLP.

Precisión sobre el nombre: la operación que se implementa es técnicamente una **correlación cruzada**, no la convolución del análisis matemático (que invierte el filtro). Como el filtro se aprende, la distinción es irrelevante en la práctica; todos los frameworks lo llaman convolución igual.

## 2. El ejemplo, paso a paso

El curso desarrolla el cálculo completo sobre un caso mínimo.

**Imagen de 4×4:**

```
35  40  41  45
40  40  46  51
43  47  49  54
48  51  55  61
```

**Filtro (kernel) de 3×3** — un detector de bordes verticales, positivo a la izquierda y negativo a la derecha:

```
 1  0  -1
 1  0  -1
 1  0  -1
```

El filtro se apoya sobre una ventana de 3×3 de la imagen, se multiplica elemento a elemento y se suman los nueve productos. Un solo número por posición.

**Posición 1** — esquina superior izquierda:

```
35×1 + 40×0 + 41×(-1) + 40×1 + 40×0 + 46×(-1) + 43×1 + 47×0 + 49×(-1)
= 35 + 0 - 41 + 40 + 0 - 46 + 43 + 0 - 49 = -18
```

**Posición 2** — un paso a la derecha:

```
40×1 + 41×0 + 45×(-1) + 40×1 + 46×0 + 51×(-1) + 47×1 + 49×0 + 54×(-1)
= 40 + 0 - 45 + 40 + 0 - 51 + 47 + 0 - 54 = -23
```

**Posición 3** — vuelve a la izquierda y baja una fila:

```
40×1 + 40×0 + 46×(-1) + 43×1 + 47×0 + 49×(-1) + 48×1 + 51×0 + 55×(-1)
= 40 + 0 - 46 + 43 + 0 - 49 + 48 + 0 - 55 = -19
```

**Posición 4** — última ventana:

```
40×1 + 46×0 + 51×(-1) + 47×1 + 49×0 + 54×(-1) + 51×1 + 55×0 + 61×(-1)
= 40 + 0 - 51 + 47 + 0 - 54 + 51 + 0 - 61 = -28
```

**Resultado, un mapa de características de 2×2:**

```
-18  -23
-19  -28
```

Dos lecturas del resultado:

- **La salida es más chica que la entrada.** De 4×4 se pasó a 2×2, porque el filtro solo entra completo en cuatro posiciones. Es exactamente el problema que resuelve el padding en la Parte 2.
- **Todos los valores son negativos.** En esta imagen la intensidad crece de izquierda a derecha, y el filtro está armado para dar positivo cuando pasa lo contrario. El signo tiene significado: dice en qué dirección va el borde, no solo que hay uno.

**Fórmula del tamaño de salida**, con entrada `N`, filtro `F`, padding `P` y stride `S`:

```
salida = (N + 2P - F) / S + 1
```

Acá: (4 + 0 - 3)/1 + 1 = **2**.

## 3. Imágenes con varios canales

La segunda mitad del bloque repite el ejercicio sobre una entrada de **4×4×3** —una imagen RGB— con un filtro de **3×3×3**.

El punto es que **la profundidad del filtro siempre iguala la de la entrada**. No se elige: si entran 3 canales, el filtro tiene 3 canales.

El procedimiento es el mismo, con un paso más al final:

1. Cada rebanada del filtro se convoluciona con su canal correspondiente de la imagen.
2. Los tres resultados **se suman entre sí** (y se le suma el sesgo).
3. Sale **un único** mapa de características de 2×2, no tres.

Esto es lo que suele confundir: un filtro de 3×3×3, por más que tenga 27 pesos repartidos en tres canales, produce **una sola** salida bidimensional. Para obtener 32 mapas de características hacen falta 32 filtros distintos, y ahí la capa tiene 32 × (3×3×3) + 32 = **896 parámetros**. Es la cuenta que se ve en el `summary()` de los notebooks de las clases 12 y 13.

> **Dos errores en las slides de esta sección.** Conviene tenerlos identificados para no perder tiempo tratando de reproducir los números:
>
> 1. El filtro del ejemplo multicanal aparece con la primera fila `1 0 1` en lugar de `1 0 -1`, que es la del ejemplo anterior.
> 2. El resultado que muestra —`-18 -23 / -19 -18`— es el del caso de **un solo canal**, y además con un `-18` donde el cálculo daba `-28`. Con tres canales idénticos y tres filtros idénticos, la suma de los tres daría el triple: `-54 -69 / -57 -84`.
>
> Lo que hay que retener del bloque es el mecanismo —sumar a lo largo de la profundidad—, no esos valores.

## 4. Relación con el resto

- La operación de esta capa es lo que reemplaza a la multiplicación matriz-vector del MLP. La segunda mitad de la red sigue siendo densa (cap. 30 del manual).
- En código: `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)` en PyTorch y `layers.Conv2D(filters, kernel_size, padding=...)` en Keras. `out_channels` / `filters` es la cantidad de filtros, es decir, cuántos mapas de características produce la capa.
- La Parte 2 completa la capa con **padding**, **stride** y la **función de activación** que se aplica al mapa de características.
