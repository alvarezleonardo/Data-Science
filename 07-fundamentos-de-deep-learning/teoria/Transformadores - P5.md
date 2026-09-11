# Transformadores — Parte 5

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 5). El PDF original está al lado.

## 1. Las conexiones residuales: un atajo alrededor de cada sub-capa

El primer diagrama de este bloque retoma la estructura del encoder de [`Transformadores - P1.md`](Transformadores%20-%20P1.md#3-estructura-interna-de-un-encoder) y le agrega el detalle que faltaba: alrededor de **cada** sub-capa (autoatención y feed forward) hay una **conexión residual** (la flecha punteada que rodea cada bloque en el diagrama) que suma la entrada de la sub-capa directamente a su salida, antes de normalizar:

```
        ↑
┌─────────────────────┐
│   Add & Normalize     │  ←── LayerNorm(entrada_de_la_subcapa + salida_de_la_subcapa)
└──────────┬────────────┘
        ↑
┌─────────────────────┐
│   Feed Forward         │
└──────────┬────────────┘
   z₁ ↑        z₂ ↑
┌─────────────────────┐
│   Add & Normalize     │  ←── LayerNorm(X + Z)
└──────────┬────────────┘
   z₁ ↑        z₂ ↑
┌─────────────────────┐
│   Self-Attention       │
└──────────┬────────────┘
   x₁ ↑        x₂ ↑
```

Concretamente, en la sub-capa de autoatención: la entrada $X$ (el embedding con codificación posicional) se suma a la salida $Z$ de la autoatención, y esa suma se normaliza:

```
LayerNorm(X + Z)
```

Y lo mismo se repite después de la red feed forward, sumando su entrada a su salida antes de normalizar otra vez.

## 2. Por qué hace falta un camino directo para el gradiente

> **Nota:** el porqué de la conexión residual no está desarrollado en el texto de la slide más allá del diagrama; esta es la ampliación que conecta la idea con lo visto antes en el módulo.

Un Transformer real apila muchas capas (6 encoders y 6 decoders en el paper original, cada uno con dos o tres sub-capas). Sin ningún mecanismo adicional, entrenar una red tan profunda expone al mismo problema que motivó buena parte de este módulo: el **gradiente desvaneciente**. Durante la retropropagación, el gradiente que llega a las primeras capas es el producto de las derivadas de todas las capas que atravesó; si cada una de esas derivadas es menor a 1, el producto se achica exponencialmente con la profundidad y las primeras capas casi no reciben señal de entrenamiento.

La conexión residual ataca esto de una forma muy directa: al sumar la entrada de la sub-capa a su salida, el gradiente tiene un **camino alternativo** para propagarse hacia atrás que no pasa por dentro de la sub-capa (por sus multiplicaciones de matrices y no linealidades), sino que la "salta" directamente por la suma. Mientras ese camino directo exista, el gradiente puede llegar a las primeras capas con una magnitud razonable incluso si el camino "largo" (a través de todas las transformaciones) se atenúa.

Esta idea es el mismo principio de fondo que ya había aparecido en el módulo, aunque con un mecanismo distinto: la [LSTM](<Memoria a Largo Plazo (LSTM).md>) (capítulo 48 del manual) resuelve el gradiente desvaneciente con su **estado de celda** $C_t$, descripto ahí como "una cinta transportadora por la que la información puede viajar muchos pasos de tiempo con modificaciones mínimas". La conexión residual del Transformer persigue el mismo objetivo — un camino casi directo para el gradiente — pero en la dimensión de **profundidad de capas** en vez de en la dimensión de **pasos de tiempo**: en la LSTM el atajo conecta $C_{t-1}$ con $C_t$ a través del tiempo; en el Transformer, la conexión residual conecta la entrada de una capa con su salida a través de la profundidad de la red.

## 3. LayerNorm: estabilizar el entrenamiento

Después de cada suma residual se aplica **LayerNorm** (normalización de capa), que reescala los valores de cada posición para que tengan media 0 y varianza 1 (sobre las dimensiones del embedding, para cada token por separado, a diferencia de la normalización por batch que promedia entre ejemplos distintos del batch).

> **Nota:** las slides no detallan la fórmula de LayerNorm ni la diferencia con BatchNorm; a los fines de este módulo alcanza con su efecto práctico: sin normalización, apilar muchas capas con conexiones residuales puede hacer que los valores crezcan sin control a medida que se van sumando las salidas de cada sub-capa (porque cada suma residual agrega más magnitud a lo que ya traía la entrada). LayerNorm evita esa deriva, manteniendo los valores en una escala estable capa tras capa, lo que en la práctica se traduce en un entrenamiento más rápido y menos sensible a la elección de la tasa de aprendizaje.

## 4. La arquitectura completa con residuales y normalización

La segunda slide muestra el diagrama de la arquitectura entera —dos encoders apilados y dos decoders apilados, más las capas finales de salida— con las conexiones residuales marcadas en cada sub-capa:

```
ENCODER #2: Self-Attention → Add&Norm → Feed Forward → Add&Norm
ENCODER #1: Self-Attention → Add&Norm → Feed Forward → Add&Norm
                                    │
                    (salida del último encoder)
                                    │
DECODER #1: Self-Attention → Add&Norm → Encoder-Decoder Attention → Add&Norm → Feed Forward → Add&Norm
DECODER #2: Self-Attention → Add&Norm → Encoder-Decoder Attention → Add&Norm → Feed Forward → Add&Norm
                                    │
                                Linear
                                    │
                                Softmax
```

Cada sub-capa —incluida la nueva **Encoder-Decoder Attention** del decoder— está envuelta en su propio `Add & Normalize`. Después del último decoder, una capa **Linear** proyecta la salida al tamaño del vocabulario y una capa **Softmax** final convierte esos valores en una distribución de probabilidad sobre las palabras posibles — el mecanismo exacto de esa última parte se desarrolla en [`Transformadores - P6.md`](Transformadores%20-%20P6.md#3-de-la-salida-del-decoder-a-una-palabra-linear--softmax).

## 5. Relación con el resto del módulo

- Las conexiones residuales y LayerNorm no son una sub-capa más entre las que ya se vieron ([`Transformadores - P2.md`](Transformadores%20-%20P2.md), [`Transformadores - P3.md`](Transformadores%20-%20P3.md)): son la "argamasa" que envuelve a cada una de ellas y permite apilarlas en profundidad sin perder la capacidad de entrenar.
- El paralelo con el estado de celda de la [LSTM](<Memoria a Largo Plazo (LSTM).md>) conecta directamente con el capítulo 48 del manual: ambos mecanismos —conexión residual y estado de celda— son formas distintas de darle al gradiente un camino que evite atravesar transformaciones no lineales completas, uno a través de las capas, el otro a través del tiempo.
- El diagrama completo de la sección 4 anticipa la estructura interna del decoder que se detalla en [`Transformadores - P6.md`](Transformadores%20-%20P6.md), incluida la capa `Linear + Softmax` final que convierte la salida del decoder en la palabra predicha.
