# Modelo Secuencia a Secuencia

> Conversión a Markdown de las slides del curso (Clase 22 — Modelo Secuencia a Secuencia, seq2seq). El PDF original está al lado.

## 1. Por qué hace falta una arquitectura nueva

Las arquitecturas recurrentes vistas hasta ahora en el módulo ([RNN](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md), [GRU](<Unidades Recurrentes con Compuertas (GRU).md>), [LSTM](<Memoria a Largo Plazo (LSTM).md>)) procesan una secuencia de entrada y producen, típicamente, una salida por cada paso de tiempo de entrada, o un único resultado final (por ejemplo, una clasificación). Eso alcanza para tareas donde la entrada y la salida tienen el mismo largo, o donde la salida es un solo valor.

> **Nota:** el motivo de fondo de esta sección (entrada y salida de largo distinto) no está explicitado en la slide con estas palabras; se agrega para justificar por qué la clase introduce una arquitectura nueva en lugar de seguir usando una RNN/GRU/LSTM sola. La slide arranca directamente definiendo el modelo.

Hay tareas de NLP donde eso no funciona: en la **traducción automática**, una oración de siete palabras en inglés puede traducirse a una oración de nueve palabras en otro idioma, y no hay ninguna garantía de que el largo coincida. Lo mismo pasa en el **resumen de texto** (un documento largo se convierte en unas pocas oraciones) o en el **subtitulado automático** de imágenes (una imagen se convierte en una oración de largo variable). Una RNN que emite una salida por cada token de entrada no puede resolver esto directamente: hace falta una arquitectura donde el largo de la salida sea independiente del largo de la entrada.

## 2. La arquitectura: encoder y decoder

El **modelo secuencia a secuencia** (*sequence-to-sequence*, seq2seq) resuelve ese problema dividiendo la red en dos partes:

- **Encoder**: toma una secuencia de entrada de longitud variable y la convierte en una representación comprimida.
- **Decoder**: toma el vector de contexto generado por el encoder y genera la secuencia de salida, también de longitud variable.

El diagrama del material muestra el mecanismo concreto:

- El **encoder** es una cadena de celdas RNN (en el dibujo, tres celdas). En cada paso $t$ recibe la entrada $x_t$ y el estado anterior $h_{t-1}$, y produce un nuevo estado $h_t = f(x_t, h_{t-1})$. Esto es exactamente la recurrencia estándar de una RNN, vista ya en la [Parte 1](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md) del módulo.
- El **último estado oculto del encoder** (tras procesar todo $x_1, x_2, x_3$) se etiqueta en el diagrama como **"Encoder Vector"**: es el **vector de contexto**, la representación comprimida de toda la secuencia de entrada.
- Ese vector de contexto alimenta al **decoder**, otra cadena de celdas RNN. En cada paso, el decoder produce una salida $y_t$ y actualiza su propio estado $s_t$ según la ecuación $s_t = g(y_{t-1}, s_{t-1}, C)$: el estado nuevo depende de la salida anterior del propio decoder, del estado anterior del decoder, y del vector de contexto $C$, que se mantiene fijo y disponible en todos los pasos del decoder.
- El decoder genera su secuencia de salida ($y_1, y_2, \dots$) paso a paso, sin que su largo tenga relación alguna con el largo de la secuencia de entrada del encoder.

| Elemento | Rol |
|---|---|
| Encoder | Comprime la secuencia de entrada de largo variable en un vector de tamaño fijo |
| Vector de contexto ($C$ / Encoder Vector) | El último estado oculto del encoder; resume toda la entrada |
| Decoder | Genera la secuencia de salida, de largo variable, a partir del vector de contexto |
| $h_t = f(x_t, h_{t-1})$ | Recurrencia del encoder |
| $s_t = g(y_{t-1}, s_{t-1}, C)$ | Recurrencia del decoder: depende de su propia salida anterior, su estado anterior y el contexto fijo |

Una segunda slide del material ilustra este mismo esquema con un ejemplo de traducción inglés-chino ("economic growth has slowed down in recent years" → "近几年经济发展变慢了"): el encoder (fila inferior) procesa la oración en inglés palabra por palabra, hasta llegar a un estado final marcado en rojo — el vector de contexto —, y ese único vector rojo se conecta (flechas punteadas) con **cada uno** de los pasos del decoder (fila superior), que va generando la oración en chino palabra por palabra, además de recibir también la salida del paso anterior del propio decoder (flechas horizontales entre los círculos del decoder).

> **Nota:** el diagrama de la traducción inglés-chino muestra al vector de contexto conectado a *todos* los pasos del decoder a la vez (no solo al primero). Es una simplificación visual habitual de este tipo de diagramas para remarcar que el contexto está disponible en cada paso de generación, coherente con la ecuación $s_t = g(y_{t-1}, s_{t-1}, C)$ de la slide anterior, donde $C$ aparece en la recurrencia de todos los pasos.

### El entrenamiento: teacher forcing

> **Nota:** esta sección es ampliación; el material no menciona el término "teacher forcing" ni describe el procedimiento de entrenamiento del decoder.

La ecuación del decoder usa $y_{t-1}$, la salida del paso anterior, como una de las entradas del paso siguiente. Durante el entrenamiento, en lugar de alimentar al decoder con lo que él mismo generó (que al principio del entrenamiento es básicamente ruido), es común alimentarlo con el valor correcto tomado del dato de entrenamiento real. Esta técnica se llama **teacher forcing**: el "maestro" (el dato real) le indica al decoder cuál era la salida correcta del paso anterior, en lugar de dejar que arrastre sus propios errores durante el aprendizaje. Esto acelera y estabiliza el entrenamiento, aunque genera una diferencia entre cómo se entrena el modelo (con la respuesta correcta disponible en cada paso) y cómo se usa en producción (donde el decoder sólo tiene sus propias predicciones anteriores, porque no existe la salida real).

## 3. El cuello de botella del vector de contexto

El material dedica una slide entera a señalar el límite central de esta arquitectura: **todo el significado de la secuencia de entrada se debe condensar en un único vector de contexto**. Esto puede ser problemático, especialmente cuando las secuencias son largas, ya que la información puede perderse o distorsionarse, haciendo que el modelo tenga dificultades para generar secuencias de salida precisas.

Dicho de otra forma: sin importar si la oración de entrada tiene cinco palabras o cincuenta, el encoder tiene que comprimir todo ese contenido en un vector del mismo tamaño fijo. Con oraciones largas, ese vector se satura: no hay espacio suficiente para representar con precisión toda la información relevante, y las partes de la entrada que quedaron más lejos en el tiempo (más al principio de la secuencia) tienden a diluirse frente a las más recientes, algo que ya se había visto como una limitación de las RNN en general.

Este cuello de botella es exactamente el problema que motiva el tema de la próxima clase: los [Mecanismos de Atención](Mecanismos%20de%20Atención.md), que reemplazan el único vector de contexto por acceso directo del decoder a todos los estados del encoder.

## 4. Relación con el resto del módulo

- Esta clase construye directamente sobre las celdas recurrentes ya vistas: el encoder y el decoder del diagrama son cadenas de RNN (el material usa el bloque genérico "RNN"; en la práctica podrían ser también celdas [GRU](<Unidades Recurrentes con Compuertas (GRU).md>) o [LSTM](<Memoria a Largo Plazo (LSTM).md>), como se ve en la clase siguiente sobre atención, donde el diagrama sí usa celdas GRU explícitamente).
- Retoma el problema de tokens y embeddings de [Procesamiento de Lenguaje Natural](Procesamiento%20de%20Lenguaje%20Natural.md): la entrada $x_t$ del encoder es, en la práctica, el embedding del token en la posición $t$.
- El límite señalado en la sección 3 (todo el significado comprimido en un solo vector de tamaño fijo) es el punto de partida explícito de la próxima clase, [Mecanismos de Atención](Mecanismos%20de%20Atención.md).
