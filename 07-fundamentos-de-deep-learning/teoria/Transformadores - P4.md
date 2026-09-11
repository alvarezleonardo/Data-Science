# Transformadores — Parte 4

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 4). El PDF original está al lado.

## 1. Por qué hace falta codificar la posición

La autoatención, tal como se definió en [`Transformadores - P2.md`](Transformadores%20-%20P2.md), calcula el score entre cada par de palabras con un producto punto que **no tiene en cuenta el orden** en que aparecen las palabras: intercambiar dos palabras de lugar en la secuencia de entrada da exactamente el mismo conjunto de scores entre pares, solo que reordenado. La autoatención es, en ese sentido, **invariante a permutaciones**.

Esto contrasta directamente con una RNN (ver [`Introducción a las redes neuronales recurrentes - P1.md`](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md)), donde el orden está implícito en la arquitectura misma: el estado oculto $h_t$ se calcula a partir de $h_{t-1}$, así que la red procesa la secuencia paso a paso y "sabe" en qué posición está simplemente por cómo fue construida la recurrencia. El Transformer, al procesar **todos los tokens en paralelo** (la ventaja de paralelización descripta en [`Transformadores - P1.md`](Transformadores%20-%20P1.md#5-por-qué-esto-importa-la-ventaja-decisiva-sobre-las-rnn)), pierde esa noción de orden como efecto colateral: si no se hace nada más, para el modelo la oración "el perro mordió al gato" sería indistinguible de "el gato mordió al perro".

> **Nota:** esta es la contracara del beneficio de paralelizar. El Transformer gana velocidad al eliminar la dependencia secuencial, pero por eso mismo necesita un mecanismo explícito y aparte para reintroducir la información de orden que la RNN traía gratis en su forma de procesar la secuencia.

## 2. La solución: sumar una codificación posicional al embedding

La solución que muestra la primera slide es simple: a cada embedding de entrada $x_i$ se le **suma** un vector de codificación posicional $t_i$, que depende únicamente de la posición $i$ del token en la secuencia (no de qué palabra sea):

```
embedding_con_posición = embedding_de_la_palabra + codificación_posicional
        x_i (con señal de tiempo)  =  x_i (embedding)  +  t_i (positional encoding)
```

En el ejemplo de la oración `Je suis étudiant`:

| Posición | Input | Embedding ($x_i$) | Positional encoding ($t_i$) | Resultado |
|---|---|---|---|---|
| 1 | Je | $x_1$ | $t_1$ | $x_1 + t_1$ |
| 2 | suis | $x_2$ | $t_2$ | $x_2 + t_2$ |
| 3 | étudiant | $x_3$ | $t_3$ | $x_3 + t_3$ |

El resultado de esa suma (`EMBEDDING WITH TIME SIGNAL`) es lo que efectivamente entra a la pila de encoders. Como la suma se hace **antes** de la primera capa, la información de posición queda mezclada con el contenido semántico del embedding, y de ahí se propaga a través de toda la red junto con el resto de la información.

## 3. Cómo se ve una codificación posicional real

La segunda slide muestra un mapa de calor de una matriz de codificación posicional real: en el eje vertical, la posición del token en la secuencia (`Token Position`, de 0 a 10); en el eje horizontal, cada una de las dimensiones del embedding (`Embedding Dimensions`, de 0 a 64); el color de cada celda representa el valor de esa dimensión para esa posición (escala de -1 a 1, con violeta para los negativos y amarillo para los positivos).

Se observan dos patrones:

- Las columnas de la izquierda (dimensiones bajas) cambian rápido de color a medida que baja la posición: oscilan con una frecuencia alta.
- Las columnas de la derecha (dimensiones altas) se ven como bandas verticales casi uniformes: cambian mucho más lento con la posición, con una frecuencia baja.

Esto corresponde a la construcción estándar de la codificación posicional del paper original, que usa funciones seno y coseno de frecuencias distintas para cada dimensión:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

donde `pos` es la posición del token y `i` el índice de la dimensión. Al variar la frecuencia con la dimensión, cada posición queda codificada con una combinación única de valores, y además el patrón tiene una propiedad útil: la codificación de una posición puede obtenerse como una función lineal de la codificación de otra, lo que en la práctica ayuda al modelo a generalizar relaciones de distancia relativa entre posiciones (aunque las slides no profundizan en esta propiedad matemática, es la motivación original de Vaswani et al. para elegir funciones periódicas en vez de, por ejemplo, simplemente numerar las posiciones con enteros crecientes).

> **Nota:** numerar las posiciones con enteros crecientes (0, 1, 2, ...) sería la alternativa más simple, pero tiene un problema práctico: los valores crecerían sin límite con secuencias largas, y el modelo tendría que generalizar a magnitudes nunca vistas en entrenamiento. Las funciones seno/coseno mantienen los valores siempre acotados entre -1 y 1, sin importar cuán larga sea la secuencia.

## 4. Relación con el resto del módulo

- Este bloque resuelve un problema que directamente **no existe** en las arquitecturas recurrentes vistas antes en el módulo ([RNN](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md), [GRU](<Unidades Recurrentes con Compuertas (GRU).md>), [LSTM](<Memoria a Largo Plazo (LSTM).md>)): en esas arquitecturas el orden viene dado por la propia recurrencia, mientras que en el Transformer hay que inyectarlo a mano porque la autoatención de [`Transformadores - P2.md`](Transformadores%20-%20P2.md) es invariante a permutaciones.
- La codificación posicional es el precio concreto que se paga por la paralelización descripta en [`Transformadores - P1.md`](Transformadores%20-%20P1.md#5-por-qué-esto-importa-la-ventaja-decisiva-sobre-las-rnn): se gana velocidad de entrenamiento (el dato del capítulo 49 del manual, ~9 minutos por época en una RNN sobre texto, es justamente el costo que el Transformer evita) a cambio de tener que resolver aparte lo que la recurrencia resolvía gratis.
- El vector resultante de esta suma (`embedding + positional encoding`) es el que entra a la primera sub-capa de autoatención de cada encoder, la misma estructura vista en [`Transformadores - P1.md`](Transformadores%20-%20P1.md#3-estructura-interna-de-un-encoder), y es también el punto de partida del diagrama de [`Transformadores - P5.md`](Transformadores%20-%20P5.md), donde se ve la suma marcada explícitamente con el símbolo ⊕.
