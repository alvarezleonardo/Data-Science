# Fundamentos de Deep Learning

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 33 | 🔄 En curso — **Módulos 1 a 4 completos** (TensorFlow, PyTorch, CNNs y RNNs) |

> Continúa [Fundamentos de redes neuronales](../06-fundamentos-de-redes-neuronales/): de `scikit-learn` se pasa a **TensorFlow** y **PyTorch**, y de ahí a CNNs, RNNs, Transformadores y modelos generativos.

## Teoría

Conversión a Markdown de las slides del curso, en [`teoria/`](teoria/):

- [`0 - Programa del módulo.md`](<teoria/0 - Programa del módulo.md>) — las 33 clases en 7 módulos
- [`Configuración del entorno.md`](<teoria/Configuración del entorno.md>) — Clase 2: instalaciones requeridas, con el equivalente para macOS y Apple Silicon

- [`Introducción a TensorFlow - Parte 1.md`](<teoria/Introducción a TensorFlow - Parte 1.md>) — Clase 3: qué es TensorFlow, tensores, grafo de cómputo, Keras
- [`Introducción a TensorFlow - Parte 2.md`](<teoria/Introducción a TensorFlow - Parte 2.md>) — Clase 3: `keras.datasets`, `Sequential` vs API funcional, `compile`
- [`Introducción a TensorFlow - Parte 3.md`](<teoria/Introducción a TensorFlow - Parte 3.md>) — Clase 3: `model.fit`, `model.evaluate` y el objeto `history`
- [`Introducción a PyTorch - Parte 1.md`](<teoria/Introducción a PyTorch - Parte 1.md>) — Clase 4: qué es PyTorch, grafo dinámico, tensores
- [`Introducción a PyTorch - Parte 2.md`](<teoria/Introducción a PyTorch - Parte 2.md>) — Clase 4: `torchvision`, transformaciones y `DataLoader`
- [`Introducción a PyTorch - Parte 3.md`](<teoria/Introducción a PyTorch - Parte 3.md>) — Clase 4: `nn.Module`, pérdidas y optimizadores
- [`Introducción a PyTorch - Parte 4.md`](<teoria/Introducción a PyTorch - Parte 4.md>) — Clase 4: el bucle de entrenamiento línea por línea y la evaluación
- [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](<teoria/01 - Introducción a las redes neuronales convolucionales - Parte 1.md>) — Clase 8: qué son las CNNs, la arquitectura completa de imagen a predicción
- [`02 - Introducción a las redes neuronales convolucionales - Parte 2.md`](<teoria/02 - Introducción a las redes neuronales convolucionales - Parte 2.md>) — Clase 8: las CNNs y la corteza visual, qué es una imagen para la computadora, canales RGB
- [`Capa convolucional - P1.md`](<teoria/Capa convolucional - P1.md>) — Clase 9: filtros, la convolución paso a paso, filtros en volumen
- [`Capa convolucional - P2.md`](<teoria/Capa convolucional - P2.md>) — Clase 9: padding, stride y ReLU, con la fórmula del tamaño de salida
- [`Capas totalmente conectadas.md`](<teoria/Capas totalmente conectadas.md>) — Clase 11: aplanamiento, capas densas, softmax y aplicaciones
- [`Introducción a las redes neuronales recurrentes - P1.md`](<teoria/Introducción a las redes neuronales recurrentes - P1.md>) — Clase 15: **arrancan las RNNs** — estado oculto, despliegue temporal, por qué tanh
- [`Introducción a las redes neuronales recurrentes - P2.md`](<teoria/Introducción a las redes neuronales recurrentes - P2.md>) — Clase 15: los cuatro tipos de RNN, funciones de activación y gradient clipping
- [`Unidades Recurrentes con Compuertas (GRU).md`](<teoria/Unidades Recurrentes con Compuertas (GRU).md>) — Clase 16: compuertas de reinicio y actualización, y por qué resuelven el gradiente desvaneciente
- [`Memoria a Largo Plazo (LSTM).md`](<teoria/Memoria a Largo Plazo (LSTM).md>) — Clase 17: las tres compuertas, el estado de celda como memoria de largo plazo, y la comparación de las tres celdas

La teoría de este módulo está consolidada en el manual: [**Parte IX — Deep Learning con frameworks**](../APUNTES-DATA-SCIENCE.md#parte-ix--deep-learning-con-frameworks), capítulos 39 a 49 (TensorFlow, PyTorch, CNNs, RNNs, GRU, LSTM y texto).

Los dos bloques enseñan **el mismo flujo en los dos frameworks**: acceso a datos → definir el modelo → configurar pérdida y optimizador → entrenar → evaluar. La comparación consolidada está en la [Parte IX del manual](../APUNTES-DATA-SCIENCE.md#parte-ix--deep-learning-con-frameworks).

## Notebooks

- [`OD_RN2_ESP_M02_S05_Implementación en PyTorch - Recurso descargable.ipynb`](<notebooks/OD_RN2_ESP_M02_S05_Implementación en PyTorch - Recurso descargable.ipynb>) — Clase 5: una red que clasifica Iris con PyTorch, del dataset a la evaluación. Recurso de clase, documentado celda por celda sin tocar el código ni los outputs. Llega a **96,67%** de exactitud.

- [`OD_RN2_ESP_M02_S06_Implementación en TensorFlow - Recurso descargable.ipynb`](<notebooks/OD_RN2_ESP_M02_S06_Implementación en TensorFlow - Recurso descargable.ipynb>) — Clase 6: **la misma red, en Keras**. Espejo exacto del anterior, pensado para comparar. Llega a **93,33%**.

- [`CNN_pytorch.ipynb`](<notebooks/CNN_pytorch.ipynb>) — Clase 12: **primera red convolucional**, sobre CIFAR-10 con PyTorch. 1.276.234 parámetros, 58,85% en 2 épocas.
- [`OD_RN2_ESP_M03_S13_CNNs_en_TensorFlow_Recurso_descargable.ipynb`](<notebooks/OD_RN2_ESP_M03_S13_CNNs_en_TensorFlow_Recurso_descargable.ipynb>) — Clase 13: **la misma CNN en Keras**. Mismos 1.276.234 parámetros, 63,03%. La diferencia es el optimizador (Adam vs SGD) y la normalización, no el framework.
- [`OD_RN2_ESP_M04_S18_RNNs en TensorFlow - Recurso descargable.ipynb`](<notebooks/OD_RN2_ESP_M04_S18_RNNs en TensorFlow - Recurso descargable.ipynb>) — Clase 18: **primera red recurrente**, sentimiento de reseñas de IMDb con LSTM. 86,64% en test.
- [`OD_RN2_ESP_M04_S19_RNNs en PyTorch - Recurso descargable.ipynb`](<notebooks/OD_RN2_ESP_M04_S19_RNNs en PyTorch - Recurso descargable.ipynb>) — Clase 19: el mismo problema en PyTorch. 74,65%, y la diferencia es `max_len` (50 contra 1.000), no el framework.
- [`mlp_iris_pytorch.ipynb`](<notebooks/mlp_iris_pytorch.ipynb>) — **notebook propio**: parte del recurso de la Clase 5 y le agrega lo que le falta — el `append` que arregla la curva de pérdida, semilla fija, seguimiento de train y test por época, matriz de confusión y detección de dispositivo (corre en **MPS**, la GPU del Mac). 96,67% con 50 épocas.

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

> **No instalar `tensorflow-metal`**: rompe la importación de TensorFlow con la versión 2.21. Detalle en [`Configuración del entorno.md`](<teoria/Configuración del entorno.md>).

Todo el procedimiento y las diferencias con las instrucciones del curso (que son para Windows) están en [`Configuración del entorno.md`](<teoria/Configuración del entorno.md>).

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
