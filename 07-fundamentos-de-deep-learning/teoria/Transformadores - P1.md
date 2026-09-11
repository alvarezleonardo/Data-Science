# Transformadores — Parte 1

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 1). El PDF original está al lado.

## 1. Qué es un Transformer

Los **Transformers** son una arquitectura de redes neuronales presentada en el artículo *"Attention is All You Need"* (Vaswani et al., 2017). Permiten manejar secuencias de datos de manera eficiente y en paralelo. Se basan completamente en **mecanismos de atención**, eliminando la necesidad de recurrir a la recurrencia (RNN, GRU, LSTM) o a la convolución para modelar relaciones secuenciales.

> **Nota:** esta eliminación de la recurrencia no es un detalle menor: es la razón central por la que el Transformer superó en la práctica a las arquitecturas recurrentes en tareas de lenguaje. La sección 5 de este archivo desarrolla por qué.

## 2. Arquitectura general: encoder-decoder

La slide muestra el diagrama clásico de traducción automática (del paper original), donde una oración en francés (`Je suis étudiant`) entra por la izquierda y sale traducida al inglés (`I am a student`) por la derecha:

```
INPUT: Je  suis  étudiant
         │
    ┌────┴────┐
    │ ENCODER │  (x6, apilados)
    │ ENCODER │
    │ ENCODER │
    │ ENCODER │
    │ ENCODER │
    │ ENCODER │
    └────┬────┘
         │  (la salida del último encoder alimenta a TODOS los decoders)
    ┌────┴────┐
    │ DECODER │  (x6, apilados)
    │ DECODER │
    │ DECODER │
    │ DECODER │
    │ DECODER │
    │ DECODER │
    └────┬────┘
         │
OUTPUT: I  am  a  student
```

La arquitectura completa tiene dos bloques apilados:

- Una **pila de encoders** (6 en el paper original), cada uno recibe la salida del encoder anterior.
- Una **pila de decoders** (también 6), cada uno recibe tanto la salida del decoder anterior como la salida del **último** encoder de la pila (esa es la flecha que en el diagrama sale del encoder superior y se abre en abanico hacia los seis decoders).

> **Nota:** el hecho de que la salida del encoder llegue a los seis decoders (no solo al primero) es la base de la **atención cruzada** (*encoder-decoder attention*) que se detalla en [`Transformadores - P6.md`](Transformadores%20-%20P6.md): cada capa del decoder puede consultar directamente la representación final de la entrada, no solo lo que produjo la capa de decoder anterior.

## 3. Estructura interna de un encoder

El segundo diagrama de este bloque abre la caja "ENCODER" y muestra sus dos sub-capas internas, apiladas una sobre otra:

```
        ↑
┌───────────────────────┐
│ Feed Forward Neural    │
│ Network                │
└───────────┬────────────┘
        ↑
┌───────────────────────┐
│ Self-Attention         │
└───────────┬────────────┘
        ↑
     (entrada)
```

- **Self-Attention** (autoatención): la sub-capa que le permite a cada palabra de la secuencia "mirar" a las demás palabras de la misma secuencia y ponderar cuánto le importa cada una para construir su propia representación. Es el mecanismo central de toda la arquitectura, desarrollado en detalle en [`Transformadores - P2.md`](Transformadores%20-%20P2.md).
- **Feed Forward Neural Network**: una red densa simple que se aplica de forma independiente a cada posición de la secuencia (la misma red, con los mismos pesos, aplicada palabra por palabra). Procesa la salida de la autoatención sin mezclar información entre posiciones distintas.

## 4. Estructura interna de un decoder

El tercer diagrama pone lado a lado un encoder y un decoder, mostrando que el decoder tiene una sub-capa adicional entre la autoatención y la red feed forward:

```
   ENCODER                        DECODER
        ↑                              ↑
┌───────────────┐          ┌───────────────────┐
│ Feed Forward   │          │ Feed Forward       │
└───────┬────────┘          └─────────┬──────────┘
    ↑                              ↑
                              ┌───────────────────┐
                              │ Encoder-Decoder     │
                              │ Attention            │
                              └─────────┬──────────┘
                                    ↑
┌───────────────┐          ┌───────────────────┐
│ Self-Attention │  ──────► │ Self-Attention      │
└───────┬────────┘          └─────────┬──────────┘
    ↑                              ↑
```

El decoder tiene entonces **tres** sub-capas en vez de dos:

1. **Self-Attention**: autoatención sobre lo que el decoder ya generó hasta el momento (con una restricción de enmascarado que se explica en [`Transformadores - P6.md`](Transformadores%20-%20P6.md#2-atención-enmascarada-masked-self-attention)).
2. **Encoder-Decoder Attention** (atención cruzada): consulta la salida del encoder para decidir en qué partes de la entrada original fijarse al generar cada palabra de la salida.
3. **Feed Forward**: igual que en el encoder, aplicada posición a posición.

La flecha horizontal entre el encoder y el decoder en el diagrama es justamente la que alimenta esa segunda sub-capa: la salida del encoder se convierte en las matrices Key y Value de la atención cruzada del decoder.

## 5. Por qué esto importa: la ventaja decisiva sobre las RNN

> **Nota:** esta sección es una ampliación que conecta el porqué de la arquitectura con lo aprendido en las clases anteriores del módulo sobre RNN, GRU y LSTM.

Una RNN (o una GRU, o una LSTM) procesa una secuencia **token por token**: el cálculo del paso $t$ necesita el estado oculto $h_{t-1}$ que salió del paso anterior. Esa dependencia secuencial impide paralelizar el cómputo dentro de una misma secuencia — por más GPU que se tenga, hay que esperar a que termine el paso 1 para arrancar el paso 2.

El Transformer rompe esa dependencia: como reemplaza la recurrencia por autoatención, **todos los tokens de la secuencia se procesan en paralelo** en cada capa. Esto es lo que en el capítulo 49 del manual (`APUNTES-DATA-SCIENCE.md`) se verificó empíricamente con el notebook de la Clase 18: entrenar una RNN sobre texto tardaba **~9 minutos por época**, un costo directamente atribuible a que la red no podía avanzar en el tiempo sin resolver primero cada paso anterior. Un Transformer del mismo tamaño de secuencia no tiene ese cuello de botella, porque ninguna posición depende del cómputo de la posición anterior dentro de la misma capa.

El costo de esa paralelización es que el Transformer, a diferencia de una RNN, no tiene ninguna noción intrínseca de orden — por eso necesita que se le inyecte artificialmente, tema del bloque de [`Transformadores - P4.md`](Transformadores%20-%20P4.md) (codificación posicional).

## 6. Relación con el resto del módulo

- Este bloque presenta el panorama completo de la arquitectura que el resto de la Clase 24 va a desarmar pieza por pieza: autoatención ([P2](Transformadores%20-%20P2.md)), múltiples cabezales ([P3](Transformadores%20-%20P3.md)), codificación posicional ([P4](Transformadores%20-%20P4.md)), normalización y residuales ([P5](Transformadores%20-%20P5.md)) y el decoder en detalle ([P6](Transformadores%20-%20P6.md)).
- El Transformer es la culminación de la línea que el módulo viene siguiendo desde las RNN clásicas: [RNN](Introducción%20a%20las%20redes%20neuronales%20recurrentes%20-%20P1.md), [GRU](<Unidades Recurrentes con Compuertas (GRU).md>) y [LSTM](<Memoria a Largo Plazo (LSTM).md>) intentaron resolver el problema del gradiente desvaneciente y de la memoria de largo plazo dentro del paradigma recurrente; el Transformer directamente abandona la recurrencia y ataca el problema con atención pura.
- El bloque previo de la clase anterior, [`Mecanismos de Atención.md`](Mecanismos%20de%20Atención.md), introduce la idea de atención aplicada sobre una RNN encoder-decoder (attention como complemento de la recurrencia); este bloque muestra el salto siguiente: la atención reemplazando por completo a la recurrencia.
