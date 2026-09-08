# Introducción a las redes neuronales convolucionales

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 01). El PDF original está al lado.

## 1. Qué son

Las **Redes Neuronales Convolucionales** (*Convolutional Neural Networks*, CNN) son un tipo de red neuronal artificial especialmente diseñada para procesar datos **utilizando un sistema que imita al ojo humano**.

Los dos puntos de la slide:

- Utilizan una operación matemática llamada **convolución**.
- Son especialmente útiles en tareas de **visión por computadora**: clasificación de imágenes, detección de objetos y reconocimiento de patrones.

## 2. La arquitectura

El diagrama del curso muestra el recorrido completo, de la imagen a la predicción. En el ejemplo entra la foto de una cebra y salen tres probabilidades: caballo 0,2 · **cebra 0,7** · perro 0,1.

| Bloque | Qué contiene | Qué hace |
|---|---|---|
| **Entrada** | la imagen | los píxeles, con sus canales de color |
| **Extracción de características** | capas de **convolución** y de **agrupamiento**, apiladas | detectan patrones cada vez más complejos, de bordes a formas |
| **Aplanamiento** | *flattening* | convierte los mapas de características en un vector |
| **Capas totalmente conectadas** | capas densas | combinan las características para decidir la clase |
| **Salida** | activación **softmax** | una probabilidad por clase, que suman 1 |

La red se parte en dos mitades con roles distintos: la primera **aprende qué mirar** y la segunda **decide**. Esa segunda mitad es exactamente el MLP del módulo anterior (§30 del manual); lo nuevo es todo lo que pasa antes.

## 3. Por qué no alcanza con un MLP

La slide no lo plantea, pero es la pregunta que justifica toda la arquitectura. Si las capas densas ya saben clasificar, ¿para qué la parte convolucional?

**El problema del tamaño.** Una imagen de 224×224 píxeles en color tiene 224 × 224 × 3 = **150.528 valores**. Conectar eso a una sola capa densa de 1.000 neuronas da más de **150 millones de pesos** en una única capa. Es inviable de entrenar y sobreajusta con cualquier cantidad de datos.

**El problema de la posición.** Un MLP recibe los píxeles como una lista plana, así que un gato en la esquina superior izquierda y el mismo gato en la esquina inferior derecha son, para la red, dos entradas completamente distintas. Tendría que aprender a reconocerlo por separado en cada posición.

La convolución resuelve los dos: aplica el **mismo filtro** a toda la imagen —así reutiliza los mismos pesos en cada posición— y detecta el patrón sin importar dónde esté.

## 4. La inspiración biológica

El material dice que las CNN "imitan al ojo humano". La referencia concreta es a la **corteza visual**, que el programa retoma en esta misma clase: las neuronas responden a estímulos de una región acotada del campo visual —su *campo receptivo*— y se organizan en capas donde cada nivel detecta patrones más complejos que el anterior.

Es la misma progresión que hace una CNN: las primeras capas reconocen bordes y colores, las intermedias combinan esos bordes en texturas y formas, y las últimas en objetos completos.

## 5. Relación con el resto del módulo

- Es la primera clase del **Módulo 3** del programa. Las siguientes desarrollan cada pieza: **capa convolucional** (filtros, padding, stride), **capa de agrupamiento** (max y average pooling, flattening) y **capas totalmente conectadas**, antes de implementarlas en PyTorch y TensorFlow.
- La segunda mitad de la red —capas densas y softmax— es lo ya visto: el MLP del módulo 06 y su implementación en los dos frameworks (caps. 40 y 41 del manual).
- La **función de activación softmax** en la salida y la pérdida de entropía cruzada categórica son las mismas del cap. 34.
