# Introducción a las redes neuronales recurrentes

> Conversión a Markdown de las slides del curso (Clase 15 — Introducción a las redes neuronales recurrentes). El PDF original está en [`material/`](material/).

## 1. Qué son

Las **Redes Neuronales Recurrentes** (*Recurrent Neural Networks*, RNN) son un tipo de red neuronal artificial diseñada para procesar **secuencias de datos**. Tienen la capacidad de analizar secuencias temporales de datos de **tamaño variable** y predecir cuál será el siguiente valor de la serie.

La idea central que las distingue de todo lo visto hasta ahora (MLP, CNN) es que tienen un **estado oculto** (*hidden state*) que se pasa de un paso temporal al siguiente. La salida en el instante $t$ no depende solo de la entrada en ese instante, sino de todo lo que la red "vio" antes, resumido en ese estado. Es lo que le permite procesar secuencias de largo variable con los mismos pesos, en lugar de necesitar una arquitectura distinta para cada longitud de entrada.

## 2. La arquitectura: la celda recurrente y su despliegue en el tiempo

El diagrama del curso muestra primero la celda con su bucle y después la misma celda **desplegada en el tiempo** (*unfold*), que es la forma más clara de entenderla: es la misma celda, reutilizada en cada paso, con sus pesos compartidos.

**Celda compacta (con el bucle):**

- Una caja `h` recibe la entrada `x` a través de una matriz de pesos `U`, y su propio estado anterior a través de una matriz `V` (el bucle).
- Produce una salida `o` a través de una matriz de pesos `W`.

**Celda desplegada (*unfold*):** la misma celda, copiada una vez por paso temporal ($t-1$, $t$, $t+1$, ...), donde el estado oculto de un paso alimenta al siguiente:

```
        o_{t-1}        o_t          o_{t+1}
          ↑W            ↑W            ↑W
 ... → h_{t-1} --V--> h_t --V--> h_{t+1} → ...
          ↑U            ↑U            ↑U
        x_{t-1}        x_t          x_{t+1}
```

Las ecuaciones que acompañan el diagrama:

$$a^{(t)} = V h^{(t-1)} + U x^{(t)} + b_u \qquad c^{(t)} = W h^{(t)} + b_v$$
$$h^{(t)} = \tanh(a^{(t)}) \qquad o^{(t)} = \text{softmax}(c^{(t)})$$

Es decir: el estado oculto nuevo combina la entrada actual (`U x^{(t)}`) con el estado anterior (`V h^{(t-1)}`), pasa por una activación **tanh**, y de ahí se calcula la salida del paso con una **softmax**.

## 3. La notación alternativa con puertas $g_1$ y $g_2$

El PDF trae una segunda representación de la misma celda, con otra notación (más cercana a la que usan Andrew Ng y varios frameworks): la entrada es $x^{\langle t \rangle}$, el estado oculto/activación es $a^{\langle t \rangle}$ (en vez de $h$) y la salida es $y^{\langle t \rangle}$. Las funciones de activación se dejan genéricas, $g_1$ y $g_2$, en lugar de fijar tanh y softmax:

$$a^{\langle t \rangle} = g_1\left(W_{aa}\, a^{\langle t-1 \rangle} + W_{ax}\, x^{\langle t \rangle} + b_a\right)$$
$$y^{\langle t \rangle} = g_2\left(W_{ya}\, a^{\langle t \rangle} + b_y\right)$$

El diagrama de esta celda muestra el mismo circuito con más detalle interno: el estado anterior $a^{\langle t-1 \rangle}$ se multiplica por $W_{aa}$, la entrada $x^{\langle t \rangle}$ por $W_{ax}$, se suman junto con el sesgo $b_a$ y pasan por $g_1$ para dar el nuevo estado $a^{\langle t \rangle}$; ese estado sigue dos caminos: continúa hacia el siguiente paso temporal, y además se multiplica por $W_{ya}$, se le suma $b_y$ y pasa por $g_2$ para dar la salida $y^{\langle t \rangle}$.

> **Nota:** las dos slides describen la misma arquitectura con dos notaciones distintas ($h$/$o$/$U,V,W$ contra $a$/$y$/$W_{aa}, W_{ax}, W_{ya}$) sin aclarar el cambio. No es un error, pero conviene tenerlo presente al comparar este material con bibliografía externa: son la misma RNN "vanilla".

## 4. Por qué tanh y no ReLU

El material no lo explicita, pero se desprende de la propia ecuación: en una RNN el estado oculto se realimenta en **cada paso temporal**, multiplicado una y otra vez por la misma matriz de pesos. Una activación no acotada como ReLU puede hacer que esos valores crezcan sin control a medida que se repite la multiplicación paso a paso (explosión numérica). **tanh**, en cambio, acota la salida al rango $(-1, 1)$ en cada paso, lo que mantiene el estado oculto en una escala estable a lo largo de toda la secuencia. Por eso las RNN clásicas usan tanh (o sigmoide) como activación del estado oculto, y reservan una activación como softmax para la salida final, donde sí conviene interpretar el resultado como probabilidades.

## 5. El límite de esta arquitectura: el gradiente desvaneciente en el tiempo

Tampoco es un punto explícito de las slides, pero es la razón de ser de las clases que siguen. Entrenar una RNN implica retropropagar el error a través de **todos los pasos temporales** desplegados (backpropagation through time). En cada paso el gradiente se multiplica una vez más, igual que en una red muy profunda (cap. 37 del manual, sobre el gradiente desvaneciente en redes profundas): con secuencias largas, esas multiplicaciones repetidas hacen que el gradiente se achique hasta casi desaparecer, y la red deja de poder aprender dependencias que están muchos pasos atrás en la secuencia.

Esta limitación es exactamente lo que motiva las arquitecturas de las próximas dos clases: **GRU** (Clase 16) y **LSTM** (Clase 17), que agregan mecanismos de compuertas para decidir qué información del estado conservar y cuál olvidar, y así sostener dependencias largas sin que el gradiente se apague.

## 6. Relación con el resto del módulo

- Es la primera clase del **Módulo 4** del programa (RNN). El propio material adelanta, sin desarrollarlo aún, que a esta introducción le siguen tipos de RNN según la forma de entrada/salida (uno a uno, uno a muchos, muchos a uno, muchos a muchos) y las funciones de activación involucradas; esos puntos no aparecen desarrollados en las 5 slides de este PDF, así que probablemente se cubran en la Parte 2 de la clase o en el material siguiente.
- La comparación natural es con el **Módulo 3** (CNN): una CNN explota la estructura **espacial** de los datos (píxeles vecinos comparten patrones), mientras que una RNN explota la estructura **temporal** (pasos vecinos de una secuencia comparten contexto). Ambas son formas de meterle al modelo una suposición razonable sobre la forma de los datos, en lugar de tratar todo como un vector plano sin orden ni estructura, como hace un MLP.
- El problema del gradiente desvaneciente conecta directamente con el cap. 37 del manual y anticipa las Clases 16 y 17 (GRU y LSTM), que son la solución práctica a la limitación que se plantea acá.
