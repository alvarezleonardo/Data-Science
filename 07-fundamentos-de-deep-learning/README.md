# Fundamentos de Deep Learning

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 33 | ✅ Aprobado (7/7) — **teoría completa**: hasta Transformadores y PLN, más el Módulo 6 (Autoencoders y GANs) |

> Continúa [Fundamentos de redes neuronales](../06-fundamentos-de-redes-neuronales/): de `scikit-learn` se pasa a **TensorFlow** y **PyTorch**, y de ahí a CNNs, RNNs, Transformadores y modelos generativos.

## Teoría

> Los **30 PDF originales** del curso están en [`teoria/material/`](teoria/material/); en `teoria/` quedan sus conversiones a Markdown.

Conversión a Markdown de las slides del curso, en [`teoria/`](teoria/):

- [`0-programa-del-modulo.md`](<teoria/0-programa-del-modulo.md>) — las 33 clases en 7 módulos
- [`configuracion-del-entorno.md`](<teoria/configuracion-del-entorno.md>) — Clase 2: instalaciones requeridas, con el equivalente para macOS y Apple Silicon

- [`introduccion-a-tensorflow-parte-1.md`](<teoria/introduccion-a-tensorflow-parte-1.md>) — Clase 3: qué es TensorFlow, tensores, grafo de cómputo, Keras
- [`introduccion-a-tensorflow-parte-2.md`](<teoria/introduccion-a-tensorflow-parte-2.md>) — Clase 3: `keras.datasets`, `Sequential` vs API funcional, `compile`
- [`introduccion-a-tensorflow-parte-3.md`](<teoria/introduccion-a-tensorflow-parte-3.md>) — Clase 3: `model.fit`, `model.evaluate` y el objeto `history`
- [`introduccion-a-pytorch-parte-1.md`](<teoria/introduccion-a-pytorch-parte-1.md>) — Clase 4: qué es PyTorch, grafo dinámico, tensores
- [`introduccion-a-pytorch-parte-2.md`](<teoria/introduccion-a-pytorch-parte-2.md>) — Clase 4: `torchvision`, transformaciones y `DataLoader`
- [`introduccion-a-pytorch-parte-3.md`](<teoria/introduccion-a-pytorch-parte-3.md>) — Clase 4: `nn.Module`, pérdidas y optimizadores
- [`introduccion-a-pytorch-parte-4.md`](<teoria/introduccion-a-pytorch-parte-4.md>) — Clase 4: el bucle de entrenamiento línea por línea y la evaluación
- [`01-introduccion-a-las-redes-neuronales-convolucionales-parte-1.md`](<teoria/01-introduccion-a-las-redes-neuronales-convolucionales-parte-1.md>) — Clase 8: qué son las CNNs, la arquitectura completa de imagen a predicción
- [`02-introduccion-a-las-redes-neuronales-convolucionales-parte-2.md`](<teoria/02-introduccion-a-las-redes-neuronales-convolucionales-parte-2.md>) — Clase 8: las CNNs y la corteza visual, qué es una imagen para la computadora, canales RGB
- [`capa-convolucional-p1.md`](<teoria/capa-convolucional-p1.md>) — Clase 9: filtros, la convolución paso a paso, filtros en volumen
- [`capa-convolucional-p2.md`](<teoria/capa-convolucional-p2.md>) — Clase 9: padding, stride y ReLU, con la fórmula del tamaño de salida
- [`capas-totalmente-conectadas.md`](<teoria/capas-totalmente-conectadas.md>) — Clase 11: aplanamiento, capas densas, softmax y aplicaciones
- [`introduccion-a-las-redes-neuronales-recurrentes-p1.md`](<teoria/introduccion-a-las-redes-neuronales-recurrentes-p1.md>) — Clase 15: **arrancan las RNNs** — estado oculto, despliegue temporal, por qué tanh
- [`introduccion-a-las-redes-neuronales-recurrentes-p2.md`](<teoria/introduccion-a-las-redes-neuronales-recurrentes-p2.md>) — Clase 15: los cuatro tipos de RNN, funciones de activación y gradient clipping
- [`unidades-recurrentes-con-compuertas-gru.md`](<teoria/unidades-recurrentes-con-compuertas-gru.md>) — Clase 16: compuertas de reinicio y actualización, y por qué resuelven el gradiente desvaneciente
- [`memoria-a-largo-plazo-lstm.md`](<teoria/memoria-a-largo-plazo-lstm.md>) — Clase 17: las tres compuertas, el estado de celda como memoria de largo plazo, y la comparación de las tres celdas
- [`procesamiento-de-lenguaje-natural.md`](<teoria/procesamiento-de-lenguaje-natural.md>) — Clase 21: tokenización por palabra, carácter y subpalabra; embeddings estáticos y contextuales
- [`modelo-secuencia-a-secuencia.md`](<teoria/modelo-secuencia-a-secuencia.md>) — Clase 22: encoder-decoder, teacher forcing y el cuello de botella del vector de contexto
- [`mecanismos-de-atencion.md`](<teoria/mecanismos-de-atencion.md>) — Clase 23: scores, softmax, contexto dinámico e interpretabilidad
- [`transformadores-p1.md`](<teoria/transformadores-p1.md>) a [`P6.md`](<teoria/transformadores-p6.md>) — Clase 24: arquitectura, autoatención con Q/K/V, múltiples cabezas, codificación posicional, residuales y normalización, y el decoder
- [`autoencoders-p1.md`](<teoria/autoencoders-p1.md>) — Clase 27, arranca el **Módulo 6**: qué es un autoencoder, estructura encoder/espacio latente/decoder
- [`autoencoders-p2.md`](<teoria/autoencoders-p2.md>) — Clase 27: tipos de autoencoders (dispersos, contractivos, de eliminación de ruido, variacionales) y aplicaciones
- [`redes-adversarias-generativas-gans.md`](<teoria/redes-adversarias-generativas-gans.md>) — Clase 28: generador vs. discriminador, el juego minimax, la pérdida no saturante, entrenamiento alternado y modos de falla (mode collapse, inestabilidad, no convergencia)

La teoría de este módulo está consolidada en el manual: [**Parte IX — Deep Learning con frameworks**](../manual/08-deep-learning-con-frameworks.md#parte-ix--deep-learning-con-frameworks), capítulos 39 a 53 (TensorFlow, PyTorch, CNNs, RNNs, GRU, LSTM, PLN y Transformadores).

Los dos bloques enseñan **el mismo flujo en los dos frameworks**: acceso a datos → definir el modelo → configurar pérdida y optimizador → entrenar → evaluar. La comparación consolidada está en la [Parte IX del manual](../manual/08-deep-learning-con-frameworks.md#parte-ix--deep-learning-con-frameworks).

## Notebooks

- [`od-rn2-esp-m02-s05-implementacion-en-pytorch-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m02-s05-implementacion-en-pytorch-recurso-descargable.ipynb>) — Clase 5: una red que clasifica Iris con PyTorch, del dataset a la evaluación. Recurso de clase, documentado celda por celda sin tocar el código ni los outputs. Llega a **96,67%** de exactitud.

- [`od-rn2-esp-m02-s06-implementacion-en-tensorflow-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m02-s06-implementacion-en-tensorflow-recurso-descargable.ipynb>) — Clase 6: **la misma red, en Keras**. Espejo exacto del anterior, pensado para comparar. Llega a **93,33%**.

- [`cnn-pytorch.ipynb`](<notebooks/cnn-pytorch.ipynb>) — Clase 12: **primera red convolucional**, sobre CIFAR-10 con PyTorch. 1.276.234 parámetros, 58,85% en 2 épocas.
- [`od-rn2-esp-m03-s13-cnns-en-tensorflow-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m03-s13-cnns-en-tensorflow-recurso-descargable.ipynb>) — Clase 13: **la misma CNN en Keras**. Mismos 1.276.234 parámetros, 63,03%. La diferencia es el optimizador (Adam vs SGD) y la normalización, no el framework.
- [`od-rn2-esp-m04-s18-rnns-en-tensorflow-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m04-s18-rnns-en-tensorflow-recurso-descargable.ipynb>) — Clase 18: **primera red recurrente**, sentimiento de reseñas de IMDb con LSTM. 86,64% en test.
- [`od-rn2-esp-m04-s19-rnns-en-pytorch-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m04-s19-rnns-en-pytorch-recurso-descargable.ipynb>) — Clase 19: el mismo problema en PyTorch. 74,65%, y la diferencia es `max_len` (50 contra 1.000), no el framework.
- [`od-rn2-esp-m05-s25-transformadores-en-python-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m05-s25-transformadores-en-python-recurso-descargable.ipynb>) — Clase 25: **un Transformer implementado desde cero** sobre el mismo problema. 82,84%, por debajo de la LSTM: entrenó 100 épocas y se sobreajustó.
- [`mlp_iris_pytorch.ipynb`](<notebooks/mlp_iris_pytorch.ipynb>) — **notebook propio**: parte del recurso de la Clase 5 y le agrega lo que le falta — el `append` que arregla la curva de pérdida, semilla fija, seguimiento de train y test por época, matriz de confusión y detección de dispositivo (corre en **MPS**, la GPU del Mac). 96,67% con 50 épocas.
- [`od-rn2-esp-m06-s29-autoencoder-en-python-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m06-s29-autoencoder-en-python-recurso-descargable.ipynb>) — Clase 29: implementación de un autoencoder.
- [`od-rn2-esp-m06-s30-redes-adversarias-generativas-en-python-recurso-descargable.ipynb`](<notebooks/od-rn2-esp-m06-s30-redes-adversarias-generativas-en-python-recurso-descargable.ipynb>) — Clase 30: implementación de una GAN.

> ⚠️ **Dos recursos de RNN tienen fallas silenciosas.** El de TensorFlow crea un `Tokenizer` nuevo al predecir en vez de usar el vocabulario de IMDb, así que las predicciones sobre frases nuevas salen invertidas pese al 86,6% de exactitud. El de PyTorch tiene el output de una sola época aunque el código pide 20. Ninguno lanza error; ambos están explicados en los propios notebooks.

> ⚠️ **El recurso de PyTorch de la Clase 5 tiene un bug.** La celda del gráfico falla con `ValueError: x and y must have same first dimension, but have shapes (10,) and (0,)`, porque el bucle de entrenamiento nunca hace `append` a `epoch_losses`. El gráfico que aparece guardado viene de otra versión del código. El arreglo —una línea— está explicado en el propio notebook.

## Entorno

Las librerías de deep learning viven en un **entorno dedicado**, `~/.venvs/dl-env`, porque TensorFlow no se puede instalar sobre el Anaconda base (conflicto con el numpy que instaló conda). Los notebooks de este módulo usan el kernel **`Python 3.12 (dl-env)`**.

| | Versión |
|---|---|
| TensorFlow | 2.21.0 (Keras 3.15.1) |
| PyTorch | 2.14.0 · **MPS disponible** |
| torchvision / torchaudio | 0.29.0 / 2.11.0 |
| numpy / pandas / matplotlib / seaborn / scikit-learn | 2.5.3 / 3.0.5 / 3.11.1 / 0.13.2 / 1.9.0 |

> **No instalar `tensorflow-metal`**: rompe la importación de TensorFlow con la versión 2.21. Detalle en [`configuracion-del-entorno.md`](<teoria/configuracion-del-entorno.md>).

Todo el procedimiento y las diferencias con las instrucciones del curso (que son para Windows) están en [`configuracion-del-entorno.md`](<teoria/configuracion-del-entorno.md>).

## Qué viene

| Módulo | Tema | Clases |
|---|---|:---:|
| 1 | Presentación | 1 |
| 2 | Introducción a TensorFlow y PyTorch | 2-7 |
| 3 | Redes Convolucionales (CNNs) | 8-14 |
| 4 | Redes Recurrentes (RNNs) | 15-20 |
| 5 | Transformadores y PLN | 21-26 |
| 6 | Autoencoders y Modelos Generativos | 27-31 |
| 7 | Cierre y evaluación integral | 32-33 |

[← Volver al índice](../README.md)
