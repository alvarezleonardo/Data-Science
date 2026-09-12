# Transformadores — Parte 3

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 3). El PDF original está en [`material/`](material/).

## 1. Por qué hace falta más de un cabezal

La autoatención vista en [`Transformadores - P2.md`](Transformadores%20-%20P2.md) usa un único juego de matrices $W^Q$, $W^K$, $W^V$. El problema es que un solo juego de matrices solo puede aprender **un tipo de relación** entre palabras a la vez. Los **múltiples cabezales de atención** (*multi-head attention*) resuelven esto corriendo varias autoatenciones en paralelo, cada una con sus propias matrices de proyección aprendidas de forma independiente.

> **Nota:** el material no explica en texto qué tipo de relación aprende cada cabeza — eso surge de la interpretabilidad empírica de los Transformers entrenados. La intuición estándar (y la que conecta con el resto del módulo) es que cada cabeza tiende a especializarse en un aspecto distinto de la relación entre palabras: una cabeza puede aprender a mirar la concordancia sintáctica (sujeto-verbo), otra relaciones semánticas (sinónimos, correferencia — a qué sustantivo se refiere un pronombre), otra patrones posicionales (mirar siempre a la palabra inmediatamente anterior). Es análogo a los **múltiples filtros de una capa convolucional** (ver [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md)): así como cada filtro de una CNN aprende a detectar un patrón visual distinto (bordes, texturas, formas) sobre la misma imagen de entrada, cada cabeza de atención aprende a detectar un tipo de relación distinto sobre la misma secuencia de entrada.

## 2. Cada cabezal, en paralelo

La primera slide muestra dos cabezales (`ATTENTION HEAD #0` y `ATTENTION HEAD #1`) operando sobre la misma entrada `X` (`Thinking Machines`):

```
                    X (Thinking Machines)
                         │
        ┌────────────────┴────────────────┐
        │                                  │
  ATTENTION HEAD #0                 ATTENTION HEAD #1
  Q₀ = X·W₀Q                        Q₁ = X·W₁Q
  K₀ = X·W₀K                        K₁ = X·W₁K
  V₀ = X·W₀V                        V₁ = X·W₁V
```

Cada cabeza tiene su propio juego de matrices de pesos ($W_0^Q, W_0^K, W_0^V$ para la cabeza 0; $W_1^Q, W_1^K, W_1^V$ para la cabeza 1), entrenadas de forma independiente. El paper original usa **8 cabezas** (`ATTENTION HEAD #0` ... `ATTENTION HEAD #7`, como marca la segunda slide), cada una produciendo su propia salida $Z_0, Z_1, \dots, Z_7$ mediante el mismo mecanismo de autoatención de [`Transformadores - P2.md`](Transformadores%20-%20P2.md#3-el-cálculo-paso-a-paso), aplicado con sus propias matrices.

## 3. Concatenar y proyectar: de 8 salidas a una sola

Una vez que las 8 cabezas produjeron sus 8 matrices de salida $Z_0 \ldots Z_7$, hace falta combinarlas en una única representación que pueda seguir viaje hacia la red feed forward. La tercera slide muestra el procedimiento en tres pasos:

1. **Concatenate all the attention heads**: se pegan una al lado de la otra las 8 matrices $Z_0, Z_1, \ldots, Z_7$, formando una matriz más ancha.
2. **Multiply with a weight matrix $W^O$**: esa concatenación se multiplica por una matriz de pesos $W^O$, entrenada junto con el resto del modelo, que comprime el resultado de vuelta al tamaño de dimensión esperado por las capas siguientes.
3. El resultado es la matriz **Z** final, que combina la información de las 8 cabezas y es la que efectivamente se envía a la red feed forward (la misma que aparecía en la estructura del encoder de [`Transformadores - P1.md`](Transformadores%20-%20P1.md#3-estructura-interna-de-un-encoder)).

```
Z = Concat(Z₀, Z₁, ..., Z₇) · Wᴼ
```

La cuarta slide muestra el diagrama completo de punta a punta: la entrada `X` se proyecta con 8 juegos distintos de $W^Q, W^K, W^V$, cada juego produce su $Q_i, K_i, V_i$ y de ahí su $Z_i$; las ocho $Z_i$ se concatenan y se multiplican por $W^O$ para dar la matriz final $Z$ (llamada $R$ en esa slide), del mismo ancho que la entrada original.

## 4. Por qué importa dividir en cabezas y no usar una sola atención más grande

> **Nota:** ampliación conceptual, coherente con la explicación estándar de la arquitectura pero no desarrollada en el texto de las slides.

Podría pensarse que en vez de 8 cabezas con dimensión más chica cada una, convendría usar una sola cabeza con la dimensión total. La diferencia práctica es que, al dividir en varias cabezas más chicas, cada una puede especializarse en un subespacio distinto de la representación sin que esa especialización compita dentro de un único cálculo de softmax. Con una sola cabeza, el softmax tendría que repartir su atención de una manera que promedia todas las relaciones relevantes en una sola distribución de pesos; con varias cabezas, cada una puede "decidir" su propia distribución de pesos de forma independiente, y recién se combinan al final, después de que cada una ya extrajo su propio patrón.

## 5. Relación con el resto del módulo

- Este bloque generaliza directamente el mecanismo de [`Transformadores - P2.md`](Transformadores%20-%20P2.md): todo lo que se explicó ahí sobre Q/K/V, el escalado por $\sqrt{d_k}$ y el softmax se aplica sin cambios dentro de cada cabeza; lo nuevo acá es correrlo 8 veces en paralelo y combinar los resultados.
- La analogía con los filtros de una CNN conecta este bloque con [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md) y [`Capa convolucional - P2.md`](Capa%20convolucional%20-%20P2.md): en ambos casos, la arquitectura multiplica la capacidad de detectar patrones distintos corriendo varias transformaciones aprendidas en paralelo sobre la misma entrada, y en ambos casos el resultado combinado (mapas de características apilados, o cabezas concatenadas) sigue viaje hacia la siguiente capa de la red.
- La matriz $Z$ resultante de este bloque es la misma matriz $Z$ que en [`Transformadores - P5.md`](Transformadores%20-%20P5.md) participa de la conexión residual y la normalización antes de llegar a la red feed forward.
