# Programa del módulo — Fundamentos de Deep Learning

> Conversión a Markdown del PDF del curso. El original está en la raíz del módulo.

## 1. Fundamentación

El objetivo del curso es proporcionar a las personas cursantes un **conocimiento avanzado y habilidades prácticas en el uso de redes neuronales** para el desarrollo de soluciones de IA complejas y de vanguardia.

## 2. Objetivos de aprendizaje

Se espera que los estudiantes puedan:

- Implementar y entrenar **redes neuronales avanzadas** utilizando **TensorFlow** y **PyTorch**.
- Aplicar **CNNs** para tareas de procesamiento de imágenes y visión por computadora.
- Utilizar **RNNs** para análisis de secuencias, procesamiento de lenguaje natural y otras aplicaciones de series temporales.
- Explorar y aplicar modelos de **Transformadores** para tareas de procesamiento de lenguaje natural de vanguardia.
- Diseñar y entrenar **Autoencoders** y **Modelos Generativos** para la generación de datos y la reducción de dimensionalidad.

## 3. Criterios de aprobación

- Realizar las actividades de **Playground** (100% de completitud).
- Aprobación de los **checkpoints de conocimiento** de cada módulo de aprendizaje.
- Aprobación del **cuestionario final** del curso.

## 4. Contenidos por módulo

### Módulo 1 — Presentación

Bienvenida al curso y test de autoevaluación de conocimientos.

- **Clase 1 — Bienvenida:** programa del curso, presentación del curso (por lo general al último), cuestionario de autoevaluación.

### Módulo 2 — Introducción a TensorFlow y PyTorch

- **Clase 2 — Configuración del entorno:** instalaciones requeridas.
- **Clase 3 — Introducción a TensorFlow:** características, fundamentos, Keras, sintaxis y flujo de trabajo.
- **Clase 4 — Introducción a PyTorch:** características, tensores, sintaxis y flujo de trabajo.
- **Clase 5 — Implementación en PyTorch:** implementación de una red neuronal en Python utilizando PyTorch.
- **Clase 6 — Implementación en TensorFlow:** implementación de una red neuronal en Python utilizando TensorFlow.
- **Clase 7 — Checkpoint de contenidos:** primera evaluación de conocimientos.

### Módulo 3 — Redes Neuronales Convolucionales (CNNs)

- **Clase 8 — Introducción a las CNNs:** introducción, arquitectura, las CNNs y la corteza visual humana.
- **Clase 9 — Capa convolucional:** filtros, padding, stride, función de activación.
- **Clase 10 — Capa de agrupamiento:** max pooling, average pooling, aplanamiento (*flattening*).
- **Clase 11 — Capas totalmente conectadas:** capas densas, aplicaciones.
- **Clase 12 — CNNs en PyTorch:** implementación en Python utilizando PyTorch.
- **Clase 13 — CNNs en TensorFlow:** implementación en Python utilizando TensorFlow.
- **Clase 14 — Checkpoint de contenidos:** segunda evaluación de conocimientos.

### Módulo 4 — Redes Neuronales Recurrentes (RNNs)

- **Clase 15 — Introducción a las RNNs:** introducción, arquitectura, tipos de RNN, funciones de activación.
- **Clase 16 — Redes GRU:** estructura y características.
- **Clase 17 — Redes LSTM:** estructura y características.
- **Clase 18 — RNNs en TensorFlow:** implementación en Python utilizando TensorFlow.
- **Clase 19 — RNNs en PyTorch:** implementación en Python utilizando PyTorch.
- **Clase 20 — Checkpoint de contenidos:** tercera evaluación de conocimientos.

### Módulo 5 — Transformadores y Procesamiento de Lenguaje Natural

- **Clase 21 — Procesamiento de Lenguaje Natural:** introducción, tokenización.
- **Clase 22 — Modelo Secuencia a Secuencia:** características, estructura encoder-decoder.
- **Clase 23 — Mecanismos de Atención:** características.
- **Clase 24 — Transformadores:** estructura general, autoatención, múltiples cabezas de atención, codificación posicional, normalización y conexiones residuales, funcionamiento del decoder.
- **Clase 25 — Transformadores en Python:** implementación en Python utilizando TensorFlow.
- **Clase 26 — Checkpoint de contenidos:** evaluación de conocimientos.

> **Nota:** el programa rotula esta evaluación como "Tercera evaluación de conocimientos", repitiendo el número de la Clase 20. Se conserva la numeración de clases tal como figura en el original y se marca la inconsistencia.

### Módulo 6 — Autoencoders y Modelos Generativos

- **Clase 27 — Autoencoders:** características, estructura, tipos de autoencoders, aplicaciones.
- **Clase 28 — Redes Adversarias Generativas (GANs):** definición, arquitectura, aplicaciones.
- **Clase 29 — Autoencoder en Python:** implementación en Python utilizando TensorFlow.
- **Clase 30 — GANs en Python:** implementación en Python utilizando TensorFlow.
- **Clase 31 — Checkpoint de contenidos:** cuarta evaluación de conocimientos.

### Módulo 7 — Cierre de curso

Cierre del curso con una evaluación integral de los conocimientos trabajados.

- **Clase 32 — Despedida:** cierre del curso.
- **Clase 33 — Evaluación integral:** evaluación integral.

## 5. Vista rápida de módulos

| Módulo | Tema central |
|---|---|
| 1 | Presentación y autoevaluación |
| 2 | Introducción a TensorFlow y PyTorch |
| 3 | Redes Neuronales Convolucionales (CNNs) |
| 4 | Redes Neuronales Recurrentes (RNNs) |
| 5 | Transformadores y PLN |
| 6 | Autoencoders y Modelos Generativos |
| 7 | Cierre de curso y evaluación integral |

## 6. Relación con el módulo anterior

Este curso continúa [Fundamentos de redes neuronales](../../06-fundamentos-de-redes-neuronales/) (módulo 06), que cubrió el perceptrón, el MLP y su implementación con `scikit-learn`. Acá se pasa a **frameworks de deep learning** —TensorFlow y PyTorch—, que permiten redes profundas, entrenamiento en GPU y arquitecturas especializadas que `scikit-learn` no cubre.
