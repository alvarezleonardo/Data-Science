# Memoria a Largo Plazo (LSTM)

> Conversión a Markdown de las slides del curso (Clase 17 — Memoria a Largo Plazo, LSTM). El PDF original está en [`material/`](material/).

## 1. Qué agrega la LSTM frente a la GRU

La [GRU](<Unidades Recurrentes con Compuertas (GRU).md>) resuelve el gradiente desvaneciente con dos compuertas y un único estado $h_t$ que se actualiza como combinación lineal entre el paso anterior y un candidato nuevo. La **LSTM** (*Long Short-Term Memory*, Hochreiter y Schmidhuber, 1997) ataca el mismo problema con una idea anterior en el tiempo pero más elaborada en la celda: en vez de una sola compuerta de mezcla, usa **tres compuertas** y, sobre todo, separa la memoria en **dos estados** distintos que viajan juntos de paso en paso:

- El **estado de celda** $C_t$ (*cell state*): la memoria de largo plazo. Es lo que le da el nombre a la arquitectura, y el material lo dibuja como una línea horizontal que atraviesa la celda casi sin tocar los bloques de cómputo intermedios: una especie de cinta transportadora por la que la información puede viajar muchos pasos de tiempo con modificaciones mínimas (solo una multiplicación y una suma por paso). Ese camino casi directo es el mecanismo concreto que combate el gradiente desvaneciente en la LSTM.
- El **estado oculto** $h_t$: lo que la celda expone hacia afuera en cada paso, análogo al $h_t$ de la RNN clásica y de la GRU, pero acá es una versión filtrada del estado de celda, no la memoria misma.

> **Nota:** aunque en este curso la LSTM se enseña después de la GRU, cronológicamente es la arquitectura más antigua: es de 1997, diecisiete años anterior a la GRU (2014). El orden pedagógico del módulo va de lo más simple (GRU, dos compuertas, un estado) a lo más elaborado (LSTM, tres compuertas, dos estados), no de lo más viejo a lo más nuevo.

## 2. Las tres compuertas

Cada compuerta de la LSTM es, igual que en la GRU, un vector de valores entre 0 y 1 producido por una sigmoide, con pesos que se aprenden durante el entrenamiento. Lo que cambia es que acá hay tres, y cada una controla un punto distinto del flujo de información entre la entrada, el estado de celda y el estado oculto:

| Compuerta | Qué decide | Sobre qué actúa |
|---|---|---|
| **Olvido** ($f_t$, *forget gate*) | Qué parte del estado de celda anterior se borra | Multiplica a $C_{t-1}$ |
| **Entrada** ($i_t$, *input gate*) | Qué parte de la información nueva (el candidato) se guarda | Multiplica al candidato $\tilde C_t$ antes de sumarlo a la celda |
| **Salida** ($o_t$, *output gate*) | Qué parte del estado de celda se expone como estado oculto | Multiplica a $\tanh(C_t)$ para producir $h_t$ |

## 3. La celda LSTM

El diagrama del material muestra la celda con el siguiente flujo (recuadro verde):

- Entran el estado de celda anterior $C_{t-1}$ y el estado oculto anterior $h_{t-1}$ (ambos por la izquierda), y la entrada actual $x_t$ (por abajo).
- $x_t$ y $h_{t-1}$ alimentan cuatro bloques de cómputo, en este orden en el dibujo: una sigmoide $\sigma$ (marcada con el número 2), otra sigmoide $\sigma$ (marcada con el número 1), un bloque tanh, y una tercera sigmoide $\sigma$ (marcada con el número 3).
- La primera sigmoide ($\sigma$, marcada (2)) es la **compuerta de olvido** $f_t$: su salida se multiplica ($\times$) directamente contra $C_{t-1}$, en el primer cruce sobre la línea horizontal del estado de celda. Así decide cuánto de la memoria anterior se conserva y cuánto se descarta.
- El bloque tanh calcula el **candidato** $\tilde C_t$: la información nueva que la celda propone agregar, a partir de $x_t$ y $h_{t-1}$.
- La segunda sigmoide ($\sigma$, marcada (1)) es la **compuerta de entrada** $i_t$: su salida se multiplica ($\times$) contra el candidato $\tilde C_t$, y ese producto se suma ($+$) al resultado de $f_t * C_{t-1}$ para formar el nuevo estado de celda $C_t$, que sigue de largo hacia la derecha (hacia el siguiente paso temporal) y también se deriva hacia abajo para calcular $h_t$.
- Sobre esa derivación, el estado de celda $C_t$ pasa por un tanh y se multiplica ($\times$) contra la salida de la tercera sigmoide ($\sigma$, marcada (3)), la **compuerta de salida** $o_t$. Ese producto es el nuevo estado oculto $h_t$, que sale hacia arriba y también continúa hacia el siguiente paso junto con $C_t$.

> **Nota:** la numeración (2), (1), (3) que usa el diagrama para las tres sigmoides no sigue el orden de izquierda a derecha en que están dibujadas (que sería olvido, entrada, salida). Los números parecen indicar el orden en que cada compuerta actúa sobre el estado de celda a lo largo del flujo (primero se olvida sobre $C_{t-1}$, después se suma la entrada nueva, por último se filtra la salida), no el orden espacial de los bloques en la slide. Es una decisión de numeración algo confusa del material, pero no una inconsistencia de contenido: la lógica de las tres compuertas es la estándar de la LSTM.

Las ecuaciones que acompañan el diagrama:

| Paso | Ecuación |
|---|---|
| Compuerta de entrada | $i_t = \sigma\left(W^{(ii)} x_t + W^{(hi)} h_{t-1}\right)$ |
| Compuerta de olvido | $f_t = \sigma\left(W^{(if)} x_t + W^{(hf)} h_{t-1}\right)$ |
| Compuerta de salida | $o_t = \sigma\left(W^{(io)} x_t + W^{(ho)} h_{t-1}\right)$ |
| Candidato (proceso de la entrada) | $\tilde C_t = \tanh\left(W^{(i\tilde C)} x_t + W^{(h\tilde C)} h_{t-1}\right)$ |
| Actualización del estado de celda | $C_t = f_t * C_{t-1} + i_t * \tilde C_t$ |
| Salida | $y_t = h_t = o_t * \tanh(C_t)$ |

> **Nota:** el orden en que la slide lista las ecuaciones (entrada, olvido, salida, candidato) tampoco coincide con el orden espacial de los bloques en el diagrama (olvido, entrada, candidato, salida) ni con el orden lógico de cómputo (primero se olvida y se calcula el candidato, después se combinan con las compuertas de entrada y salida). Es una lista de definiciones, no una secuencia de pasos; las fórmulas en sí son consistentes entre diagrama y ecuaciones, a diferencia de lo que pasaba en la slide de la GRU con el reset gate.

## 4. Comparación de las tres celdas (RNN, GRU, LSTM)

El material incluye una slide que pone lado a lado los diagramas de las tres celdas vistas en el módulo. Confirma visualmente la progresión:

| | RNN | GRU | LSTM |
|---|---|---|---|
| Compuertas | Ninguna | 2 (reinicio, actualización) | 3 (olvido, entrada, salida) |
| Estados que viajan entre pasos | Uno, $h_t$ | Uno, $h_t$ | Dos, $h_t$ y $C_t$ |
| Mecanismo de actualización | Todo se mezcla con una tanh en cada paso | Combinación lineal entre $h_{t-1}$ y el candidato, ponderada por $z_t$ | Suma de dos productos ($f_t * C_{t-1} + i_t * \tilde C_t$) sobre un estado de celda separado |
| Camino para el gradiente | Ninguno directo: se multiplica por la derivada de la tanh en cada paso | Directo cuando $z_t \to 0$: $h_t \approx h_{t-1}$ | Directo casi permanente: $C_t$ se propaga con solo una multiplicación y una suma por paso |

Esta tabla amplía la comparación GRU-LSTM que ya se había adelantado en la clase anterior: la diferencia central sigue siendo que la LSTM separa memoria de largo plazo (estado de celda) y salida expuesta (estado oculto), mientras que la GRU los mantiene fusionados en un solo $h_t$. Esa separación es lo que suele darle a la LSTM una ventaja en secuencias muy largas, al costo de más parámetros y de un entrenamiento más lento por celda.

## 5. Relación con el resto del módulo

- Esta clase cierra la familia de arquitecturas con compuertas que arrancó con la [GRU](<Unidades Recurrentes con Compuertas (GRU).md>): ambas nacen para resolver el mismo problema, el gradiente desvaneciente de la [RNN clásica](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md#5-el-límite-de-esta-arquitectura-el-gradiente-desvaneciente-en-el-tiempo), pero la LSTM lo hace con un mecanismo más elaborado: tres compuertas en vez de dos, y un estado de celda dedicado en vez de una única mezcla.
- El estado de celda de la LSTM es la versión más explícita, dentro del módulo, de la idea de "camino directo para el gradiente" que ya había aparecido de forma más limitada en la GRU (cuando $z_t$ es chico y $h_t \approx h_{t-1}$).
- El material incluye a continuación una slide sobre **RNN Bidireccionales (BRNN)**, que no es una arquitectura de compuertas sino una forma distinta de conectar celdas (RNN, GRU o LSTM) para que cada paso vea contexto tanto pasado como futuro de la secuencia. No forma parte del contenido de esta clase sobre LSTM; queda para tratarse en el tema que sigue del módulo.
