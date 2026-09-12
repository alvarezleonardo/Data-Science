# Unidades Recurrentes con Compuertas (GRU)

> Conversión a Markdown de las slides del curso (Clase 16 — Unidades Recurrentes con Compuertas, GRU). El PDF original está en [`material/`](material/).

## 1. El problema que resuelven las compuertas

La RNN clásica, vista en la clase anterior, tiene un único mecanismo para actualizar el estado oculto en cada paso: combinar toda la entrada nueva con todo el estado anterior y pasar el resultado por una tanh. No hay forma de decirle a la red "conservá este dato varios pasos" o "ignorá esta parte del pasado": todo se mezcla por igual en cada paso, y por eso el gradiente se desvanece en secuencias largas (ver [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md#5-el-límite-de-esta-arquitectura-el-gradiente-desvaneciente-en-el-tiempo)).

La **GRU** (*Gated Recurrent Unit*) introduce **compuertas**: un valor entre 0 y 1, producido por una sigmoide, que decide cuánta información pasa por un camino determinado. Una compuerta en 0 bloquea el paso, una compuerta en 1 lo deja pasar completo, y los valores intermedios dejan pasar una fracción. Es un mecanismo de control aprendido: la red ajusta los pesos de la sigmoide durante el entrenamiento para decidir, en cada paso y para cada dimensión del estado, cuánto conservar y cuánto reemplazar.

La GRU tiene dos compuertas:

- La **compuerta de actualización** ($z_t$, *update gate*) decide cuánto del estado anterior se conserva y cuánto del candidato a nuevo estado entra.
- La **compuerta de reinicio** ($r_t$, *reset gate*) decide cuánto del estado anterior se ignora al calcular ese candidato.

## 2. La celda GRU

El diagrama del material muestra la celda con el siguiente flujo (recuadro verde):

- Entran el estado anterior $h_{t-1}$ (por arriba, cruzando todo el recuadro) y la entrada actual $x_t$ (por abajo).
- $x_t$ y $h_{t-1}$ alimentan tres bloques de cómputo: una sigmoide $\sigma$ para la compuerta de reinicio $r_t$, otra sigmoide $\sigma$ para la compuerta de actualización $z_t$, y una tanh para el candidato $\tilde h_t$.
- La compuerta de reinicio $r_t$ se multiplica ($\times$) contra $h_{t-1}$ antes de que ese resultado entre al bloque tanh: así es como $r_t$ controla cuánto del pasado se usa para calcular el candidato.
- La compuerta de actualización $z_t$ se usa dos veces, de forma complementaria: directamente como factor del candidato $\tilde h_t$, y como $1 - z_t$ (círculo rosa "1-" en el diagrama) como factor de $h_{t-1}$.
- El estado anterior $h_{t-1}$, multiplicado por $(1 - z_t)$, se suma ($+$) con el candidato $\tilde h_t$, multiplicado por $z_t$, para dar el nuevo estado $h_t$, que sale hacia arriba y continúa hacia el siguiente paso temporal.

Las ecuaciones que acompañan el diagrama:

| Paso | Ecuación |
|---|---|
| Compuerta de reinicio | $r_t = \sigma\left(W^{(ir)} x_t + W^{(hr)} h_{t-1}\right)$ |
| Compuerta de actualización | $z_t = \sigma\left(W^{(iz)} x_t + W^{(hz)} h_{t-1}\right)$ |
| Candidato (proceso de la entrada) | $\tilde h_t = \tanh\left(W^{(i\tilde h)} x_t + W^{(h\tilde h)} h_{t-1}\right)$ |
| Actualización del estado oculto | $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde h_t$ |
| Salida | $y_t = h_t$ |

> **Nota:** en la ecuación del candidato $\tilde h_t$, el diagrama de flujo muestra que el estado que entra al bloque tanh ya viene multiplicado por $r_t$ (es $r_t * h_{t-1}$, no $h_{t-1}$ sin modificar). La fórmula escrita en la slide, sin embargo, usa $W^{(h\tilde h)} h_{t-1}$ y no menciona $r_t$ explícitamente. Es la formulación estándar de la GRU la que incluye el reset gate multiplicando a $h_{t-1}$ dentro del candidato ($\tilde h_t = \tanh(W^{(i\tilde h)} x_t + r_t * (W^{(h\tilde h)} h_{t-1}))$); el diagrama es consistente con esa formulación estándar, pero la tabla de ecuaciones del PDF omite el término $r_t$. Es una inconsistencia entre el diagrama y la fórmula escrita en la propia slide, no un cambio de notación deliberado.

En la ecuación de $z_t$, la slide usa $\bar{x}_t$ (con la barra) en lugar de $x_t$; es la misma variable de entrada, probablemente una inconsistencia tipográfica del material.

## 3. Por qué esto resuelve el gradiente desvaneciente

La ecuación clave es la actualización del estado: $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde h_t$. Es una **combinación lineal** entre el estado anterior y el candidato nuevo, ponderada por la compuerta $z_t$.

Cuando $z_t$ es cercano a 0, $h_t \approx h_{t-1}$: el estado prácticamente no cambia, se copia casi tal cual de un paso al siguiente. Eso es justo lo que crea el **camino más directo** para el gradiente: al derivar $h_t$ respecto de $h_{t-1}$, aparece un término $(1 - z_t)$ que puede ser cercano a 1, en lugar de depender pura y exclusivamente de la derivada de una tanh (que en la RNN clásica multiplica el gradiente en cada paso y lo va achicando). Con ese camino, el gradiente puede fluir hacia atrás muchos pasos sin verse forzado a multiplicarse por derivadas chicas en cada uno, y así la red puede aprender dependencias que están lejos en la secuencia sin que el gradiente se apague.

## 4. GRU frente a LSTM

El material no desarrolla esta comparación en las slides (la GRU es el foco exclusivo de esta clase), pero es la referencia obligada para ubicar a la GRU dentro de la familia de RNN con compuertas:

| | GRU | LSTM |
|---|---|---|
| Compuertas | 2 (actualización, reinicio) | 3 (entrada, olvido, salida) |
| Estado | Uno solo, $h_t$ | Dos: el estado oculto $h_t$ y un estado de celda $C_t$ separado |
| Parámetros | Menos | Más |
| Velocidad de entrenamiento | Más rápida (menos parámetros por celda) | Más lenta |
| Rendimiento en secuencias muy largas | Bueno | Suele ser algo mejor, por el estado de celda dedicado |

> **Nota:** esta tabla no proviene del PDF de la clase; se agrega como contexto porque el propio material anticipa la LSTM como siguiente tema del módulo.

## 5. Relación con el resto del módulo

- Esta clase retoma directamente el problema planteado al cierre de la [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md#5-el-límite-de-esta-arquitectura-el-gradiente-desvaneciente-en-el-tiempo): la RNN clásica no puede sostener dependencias largas porque el gradiente se desvanece. La GRU es la primera solución de la familia de arquitecturas con compuertas.
- La [Parte 2](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P2.md) de la introducción a RNN ya había anticipado, con el gradient clipping, una técnica para el problema inverso (explosión del gradiente) que actúa sobre el entrenamiento; la GRU en cambio ataca el desvanecimiento desde el diseño mismo de la celda.
- El siguiente paso natural del módulo es la **LSTM**, que generaliza la idea de compuertas con un mecanismo más elaborado (tres compuertas y un estado de celda separado del estado oculto).
