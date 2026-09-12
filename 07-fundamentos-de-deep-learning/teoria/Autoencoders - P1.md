# Autoencoders - P1

> Conversión a Markdown de las slides del curso (Clase 27 — Autoencoders y Modelos Generativos, arranca el **Módulo 6**). El PDF original está en [`material/`](<material/Autoencoders - P1.pdf>).

## 1. Qué es un autoencoder

Un **autoencoder** es un tipo de arquitectura de red neuronal que se usa para **aprendizaje no supervisado**: no recibe etiquetas, solo datos de entrada, y su objetivo es aprender a reconstruirlos a sí mismo. El material lo resume en dos verbos:

- **Comprime (codifica)** los datos de entrada hasta reducirlos a sus características esenciales.
- **Reconstruye (decodifica)** la entrada original a partir de esa representación comprimida.

A esa representación comprimida el material la llama **espacio de representación** o, más adelante, **espacio latente**: un conjunto de **variables latentes** que capturan lo esencial de los datos de entrada con muchas menos dimensiones que la entrada original.

> **Nota:** el material no explica **por qué** obligar a la red a pasar por ese cuello de botella (el espacio latente) sirve para algo. La razón, que no está en la slide, es que si la red pudiera copiar la entrada directamente a la salida sin comprimirla, no aprendería nada útil — la tarea de reconstrucción sería trivial. Al forzar los datos a pasar por una representación más chica que la entrada, la red se ve obligada a descartar redundancia y quedarse solo con la estructura que le permite reconstruir lo más importante, que es justamente lo que se busca capturar como "variables latentes".

## 2. Estructura: encoder, espacio latente, decoder

El material presenta el diagrama de bloques de la arquitectura, con datos de entrada a la izquierda y datos de entrada reconstruidos a la derecha:

```
X (datos de entrada) → [Encoder] → Espacio latente → [Decoder] → Y (datos reconstruidos)
```

- **Encoder**: toma la entrada `X` y la comprime en un espacio de representación más pequeño y denso (el espacio latente).
- **Espacio latente**: el "cuello de botella" de la arquitectura — el vector de variables latentes, de dimensión mucho menor que `X`.
- **Decoder**: toma esa representación comprimida y la descomprime en una reproducción de la entrada original, `Y`.

El objetivo de entrenamiento es que `Y` se parezca lo más posible a `X` (la red se entrena minimizando algún error de reconstrucción entre ambos, por ejemplo el error cuadrático medio), aunque el material no explicita la función de pérdida en esta parte del contenido.

> **Nota:** encoder y decoder suelen ser, en la práctica, redes neuronales "espejadas": si el encoder va reduciendo dimensiones capa a capa (por ejemplo 784 → 128 → 32), el decoder hace el camino inverso (32 → 128 → 784) para devolver una salida del mismo tamaño que la entrada original. El material no muestra esta simetría explícitamente en el diagrama, pero es la forma habitual de implementarlos.

## 3. Relación con el resto del módulo

- Esta clase (27) es la primera del **Módulo 6 — Autoencoders y Modelos Generativos**, que cierra el bloque de contenidos nuevos del curso antes del módulo final de cierre y evaluación integral.
- A diferencia de los bloques anteriores del módulo (CNNs, RNNs, Transformadores), que resuelven tareas **supervisadas** (clasificación, traducción, predicción de la siguiente palabra), los autoencoders introducen el **aprendizaje no supervisado** dentro de deep learning: no hay etiqueta objetivo distinta de la propia entrada.
- El concepto de **espacio latente** que se introduce acá es la base para entender, más adelante en el módulo, los **autoencoders variacionales** (que aprenden una distribución de probabilidad sobre ese espacio en lugar de un punto fijo) y los modelos generativos en general — ver [Autoencoders - P2](<Autoencoders - P2.md>).
