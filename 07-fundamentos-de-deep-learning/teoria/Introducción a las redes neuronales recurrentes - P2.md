# Introducción a las redes neuronales recurrentes (Parte 2)

> Conversión a Markdown de las slides del curso (Clase 15 — Introducción a las redes neuronales recurrentes, Parte 2). El PDF original está al lado.

## 1. RNN frente a otras arquitecturas

Una Red Neuronal Convolucional, bien entrenada, clasifica con alta precisión una imagen o una palabra tomadas de forma individual. La pregunta que abre esta Parte 2 es qué pasa cuando, en lugar de un dato suelto, se le presenta a la red una **secuencia**: un video (secuencia de imágenes) o una conversación (secuencia de palabras).

Ese tipo de arquitecturas (CNN, MLP) no tiene forma de analizar la relación entre los elementos de la secuencia: cada imagen o palabra se procesa de manera aislada, sin memoria de lo que vino antes.

Las RNN no tienen ese problema porque están diseñadas justamente para eso: analizar secuencias. El ejemplo que usa el material es elocuente: la palabra "recurrente" está formada por caracteres que, tomados de a uno o en un orden distinto, no significan nada — el significado está en el orden y en la relación entre ellos. Además, a diferencia de una entrada de tamaño fijo, una secuencia **no tiene un tamaño predefinido**: puede tener 3 palabras o 300.

## 2. Notación de esta sección

Los diagramas de "Tipos de RNN" usan la notación de Andrew Ng ya introducida en la [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md#3-la-notación-alternativa-con-puertas-g_1-y-g_2): el estado oculto es $a^{\langle t \rangle}$, la entrada $x^{\langle t \rangle}$ y la salida $\hat{y}^{\langle t \rangle}$, con $a^{\langle 0 \rangle}$ como estado inicial. $T_x$ es la longitud de la secuencia de entrada y $T_y$ la de la secuencia de salida.

## 3. Tipos de RNN

Según cómo se relacionan las longitudes de entrada ($T_x$) y salida ($T_y$), hay cuatro arquitecturas posibles de RNN. Todas comparten la misma celda recurrente; lo que cambia es cuántas veces se lee la entrada y cuántas veces se produce salida.

| Tipo | Relación $T_x$/$T_y$ | Ejemplo del material |
|---|---|---|
| One-to-one | $T_x = T_y = 1$ | Caso degenerado: una entrada, una salida, sin recurrencia real (equivale a una red feedforward clásica). |
| One-to-many | $T_x = 1$, $T_y > 1$ | Generación de descripciones de imagen (*image captioning*): a partir de una imagen de un obrero con chaqueta naranja, la RNN genera la secuencia de palabras "Obrero con chaqueta de color naranja realizando un trabajo en la vía". |
| Many-to-one | $T_x > 1$, $T_y = 1$ | Análisis de sentimiento: a partir del texto de una reseña de cine (que "aporta nuevos elementos a la fórmula tradicional..."), la RNN produce una única salida, la calificación en estrellas. |
| Many-to-many | $T_x = T_y$ o $T_x \neq T_y$ | Traducción automática: una conversación profesor-alumno en inglés se traduce, palabra por palabra o como secuencia completa, al español. |

**One-to-one** ($T_x = T_y = 1$):

```
        ŷ
        ↑
        [a]
        ↑
        x
```

Una sola celda, sin desplegar en el tiempo: no hay secuencia real, es el caso límite.

**One-to-many** ($T_x = 1$, $T_y > 1$): una única entrada $x$ alimenta la primera celda junto con el estado inicial $a^{\langle 0 \rangle}$, y a partir de ahí la red se despliega en el tiempo generando una salida $\hat{y}^{\langle 1 \rangle}, \hat{y}^{\langle 2 \rangle}, \ldots, \hat{y}^{\langle T_y \rangle}$ por paso, encadenando el estado oculto de una celda a la siguiente:

```
        ŷ<1>      ŷ<2>            ŷ<Ty>
         ↑          ↑                ↑
a<0> → [a] → [a] → ... → [a]
         ↑
         x
```

**Many-to-one** ($T_x > 1$, $T_y = 1$): la secuencia de entrada $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \ldots, x^{\langle T_x \rangle}$ se procesa paso a paso, y recién al final, sobre el último estado oculto, se produce la única salida $\hat{y}$:

```
a<0> → [a] → [a] → ... → [a] → ŷ
         ↑     ↑            ↑
        x<1>  x<2>         x<Tx>
```

**Many-to-many**: hay dos variantes.

- $T_x = T_y$: una salida por cada entrada, en el mismo paso temporal (cada celda produce su $\hat{y}^{\langle t \rangle}$ a medida que procesa su $x^{\langle t \rangle}$). Es el caso de etiquetado palabra por palabra.
- $T_x \neq T_y$: las longitudes de entrada y salida difieren (típico de traducción, donde una oración en un idioma no tiene por qué tener la misma cantidad de palabras que su traducción). El diagrama del material no desarrolla esta variante en detalle más allá de la fórmula $T_x \neq T_y$; en la práctica se resuelve con una arquitectura encoder-decoder, que procesa toda la entrada antes de empezar a generar la salida.

> **Nota:** el PDF no explicita la arquitectura encoder-decoder para el caso $T_x \neq T_y$; solo presenta la fórmula. Se menciona acá porque es la solución estándar y puede aparecer en clases posteriores del módulo.

## 4. Funciones de activación

El material cierra con una tabla comparativa de las tres funciones de activación relevantes para RNN:

| Función | Fórmula | Rango |
|---|---|---|
| Sigmoid | $g(z) = \dfrac{1}{1+e^{-z}}$ | $(0, 1)$ |
| Tanh | $g(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ |
| ReLU | $g(z) = \max(0, z)$ | $[0, +\infty)$ |

Como ya se explicó en la [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md#4-por-qué-tanh-y-no-relu), en el estado oculto de una RNN conviene una activación acotada (sigmoid o tanh) porque el valor se realimenta en cada paso temporal, y una activación sin cota como ReLU puede hacer que la magnitud crezca sin control a lo largo de la secuencia.

El material agrega una segunda gráfica relacionada con el entrenamiento: el **gradient clipping** (recorte de gradiente). Se grafica la norma del gradiente después del recorte, $\lVert \nabla \mathcal{L} \rVert_{\text{clipped}}$, en función de la norma original, $\lVert \nabla \mathcal{L} \rVert$: mientras la norma del gradiente es menor a un umbral $C$, se deja pasar sin cambios (la recta a 45°); una vez que la supera, se la recorta al valor $C$ (la meseta horizontal). Es una técnica de entrenamiento — no cambia la arquitectura de la celda — que se usa junto con las RNN para evitar el problema inverso al gradiente desvaneciente: la **explosión del gradiente**, cuando las multiplicaciones repetidas a lo largo de los pasos temporales hacen que el gradiente crezca en vez de achicarse.

> **Nota:** el gráfico de gradient clipping aparece sin rótulo de sección propia, pegado debajo de la tabla de funciones de activación en la misma slide. Se lo trata acá como un punto aparte porque conceptualmente no es una función de activación, sino una técnica de estabilización del entrenamiento.

## 5. Relación con el resto del módulo

- Esta Parte 2 completa lo que la [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md) dejaba anunciado: los tipos de RNN y las funciones de activación que no llegaba a desarrollar.
- Los cuatro tipos de RNN (one-to-one, one-to-many, many-to-one, many-to-many) son ortogonales al problema del gradiente desvaneciente explicado en la Parte 1: cualquiera de estas arquitecturas, si la secuencia es larga, sufre la misma dificultad para propagar el gradiente hacia atrás en el tiempo.
- El gradient clipping mitiga la explosión del gradiente, pero no resuelve el desvanecimiento — ese es el problema que motiva específicamente las arquitecturas de compuertas de las próximas clases: [GRU](Unidades%20Recurrentes%20con%20Compuertas%20%28GRU%29.md) y LSTM.
