# Transformadores — Parte 6

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 6). El PDF original está en [`material/`](material/).

## 1. El decoder genera la salida un token a la vez

A diferencia del encoder, que procesa toda la secuencia de entrada de una sola pasada, el **decoder** genera la secuencia de salida de forma **autorregresiva**: produce un token, lo agrega a lo ya generado, y usa esa salida ampliada como entrada para producir el siguiente token. Las tres slides de este bloque muestran ese proceso en distintos pasos de tiempo (`Decoding time step`).

**Paso 1** (primera slide): con la oración de entrada ya procesada por la pila de encoders (`Je suis étudiant`), el decoder todavía no tiene ninguna salida previa. Genera el primer token de la traducción (`I`, aunque en esta slide en particular el output todavía no se muestra completo).

**Paso 2** (segunda slide): ahora el decoder recibe como entrada la palabra generada en el paso anterior (`PREVIOUS OUTPUTS: I`), la embebe y le suma su codificación posicional, exactamente igual que se hace con la entrada del encoder (ver [`Transformadores - P4.md`](Transformadores%20-%20P4.md#2-la-solución-sumar-una-codificación-posicional-al-embedding)). Con esa entrada, el decoder produce el segundo token de salida.

```
Paso 1: ENCODERS(Je, suis, étudiant) → DECODERS()        → "I"
Paso 2: ENCODERS(Je, suis, étudiant) → DECODERS("I")     → "am"
Paso 3: ENCODERS(Je, suis, étudiant) → DECODERS("I am")  → "a"
   ...
```

Esta es la razón estructural por la que el decoder, a diferencia del encoder, no puede procesar toda la secuencia de salida en una sola pasada durante la generación: cada token nuevo depende de los tokens que el propio decoder generó antes.

> **Nota:** esto reintroduce, solo durante la generación (inferencia), una forma de secuencialidad parecida a la de una RNN — el decoder no puede saltarse pasos porque necesita lo que generó antes. La diferencia con una RNN es que durante el **entrenamiento** sí se puede paralelizar (con *teacher forcing*, alimentando de una sola vez toda la secuencia objetivo real en paralelo, en vez de generarla token a token), algo que una RNN entrenando sobre secuencias no puede hacer de la misma manera porque su propio cómputo interno es secuencial paso a paso. En inferencia, en cambio, el Transformer sí pierde esa ventaja de paralelización: genera token por token igual que lo haría una RNN.

## 2. Atención enmascarada (Masked Self-Attention)

> **Nota:** el material de estas slides no incluye un diagrama específico de la máscara de atención, pero el mecanismo es imprescindible para que el proceso autorregresivo de la sección 1 sea consistente, y está anticipado en el temario de la clase ("funcionamiento del decoder"). Se documenta acá como ampliación necesaria.

La primera sub-capa del decoder (vista en la estructura de [`Transformadores - P1.md`](Transformadores%20-%20P1.md#4-estructura-interna-de-un-decoder)) es una autoatención igual a la del encoder, pero con una restricción: cada posición **solo puede atender a las posiciones anteriores o a sí misma, nunca a las posiciones futuras**. Esto se llama atención **enmascarada** porque, en la práctica, se implementa poniendo en $-\infty$ (antes del softmax) los scores correspondientes a posiciones futuras, de modo que después del softmax esas posiciones reciben peso 0.

La necesidad de esta máscara es directa: durante el entrenamiento, al decoder se le puede alimentar de una sola vez toda la secuencia objetivo (para paralelizar, como se explicó en la sección 1), pero si al calcular la salida en la posición 2 la autoatención pudiera "ver" la posición 3, el modelo estaría haciendo trampa: usaría información del futuro que en el momento real de generación (inferencia, token a token) todavía no existiría. La máscara garantiza que el entrenamiento sea fiel a las condiciones reales de generación, donde el token en la posición $t$ nunca puede depender de tokens que se generan después.

## 3. Atención cruzada (Encoder-Decoder Attention)

La segunda sub-capa del decoder es la **atención cruzada**, visible en la segunda slide de este bloque con las matrices $K_{encdec}$ y $V_{encdec}$: a diferencia de la autoatención (donde Q, K y V salen todos de la misma secuencia), acá:

- La **Query** sale del propio decoder (de la salida de su autoatención enmascarada, ya normalizada).
- La **Key** y el **Value** salen de la salida del **encoder** (la representación final de la oración de entrada, la misma que en [`Transformadores - P1.md`](Transformadores%20-%20P1.md#2-arquitectura-general-encoder-decoder) se mostraba llegando a los seis decoders).

```
Autoatención del decoder:    Q, K, V  →  todos del decoder (con máscara)
Atención cruzada:            Q        →  del decoder
                              K, V     →  del encoder
```

Esto es lo que le permite al decoder, en cada paso de generación, consultar directamente qué partes de la oración de entrada son relevantes para decidir la próxima palabra de la salida — es el mecanismo que reemplaza, de forma mucho más flexible, a lo que la atención clásica sobre RNN (repasada en [`Transformadores - P2.md`](Transformadores%20-%20P2.md#1-autoatención-self-attention)) hacía entre un decoder RNN y un encoder RNN.

## 4. De la salida del decoder a una palabra: Linear + Softmax

La tercera slide muestra el último tramo, después de que la pila completa de decoders (con sus autoatenciones enmascaradas, atenciones cruzadas y redes feed forward, cada una con su `Add & Normalize` como en [`Transformadores - P5.md`](Transformadores%20-%20P5.md)) produjo su vector de salida final para la posición actual:

1. **Linear**: una capa densa proyecta ese vector al tamaño del vocabulario completo (`vocab_size`), dando un score (un "logit") para cada palabra posible del vocabulario.
2. **Softmax**: esos scores se convierten en una distribución de probabilidad sobre el vocabulario.
3. Se elige la palabra con mayor probabilidad (en el ejemplo, la posición `5` del vocabulario, resaltada en amarillo, corresponde a la palabra `am`).

```
Decoder stack output → Linear (proyecta a vocab_size) → Softmax (probabilidades) → argmax → palabra
```

Esa palabra elegida es la que se retroalimenta como `PREVIOUS OUTPUTS` para generar el siguiente token, cerrando el ciclo autorregresivo descripto en la sección 1.

## 5. Relación con el resto del módulo

- Este bloque cierra la Clase 24 completando el mecanismo del decoder que había quedado esbozado en [`Transformadores - P1.md`](Transformadores%20-%20P1.md#4-estructura-interna-de-un-decoder): las tres sub-capas (autoatención enmascarada, atención cruzada, feed forward) reutilizan exactamente el mecanismo de atención de [`Transformadores - P2.md`](Transformadores%20-%20P2.md) y [`Transformadores - P3.md`](Transformadores%20-%20P3.md), con las variantes de qué secuencia aporta Q y cuál aporta K/V, y con la restricción de enmascarado.
- El proceso autorregresivo de generación palabra por palabra tiene un paralelo directo con cómo un decoder RNN genera texto en las arquitecturas recurrentes del módulo (ver la mención a generación de texto en [`Introducción a las redes neuronales recurrentes - P2.md`](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P2.md)): en ambos casos, en inferencia, cada palabra nueva depende de las anteriores y no hay forma de evitar esa secuencialidad — la diferencia central del Transformer está en el **entrenamiento**, donde sí logra paralelizar gracias a la atención enmascarada en vez de a una recurrencia real.
- Con este bloque se completa el recorrido de la Clase 24: arquitectura general ([P1](Transformadores%20-%20P1.md)), autoatención ([P2](Transformadores%20-%20P2.md)), múltiples cabezales ([P3](Transformadores%20-%20P3.md)), codificación posicional ([P4](Transformadores%20-%20P4.md)), normalización y residuales ([P5](Transformadores%20-%20P5.md)) y, acá, el funcionamiento completo del decoder.
