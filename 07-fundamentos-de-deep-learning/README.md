# Fundamentos de Deep Learning

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 33 | 🔄 En curso — entorno instalado · material de las Clases 3 y 4 recibido |

> Continúa [Fundamentos de redes neuronales](../06-fundamentos-de-redes-neuronales/): de `scikit-learn` se pasa a **TensorFlow** y **PyTorch**, y de ahí a CNNs, RNNs, Transformadores y modelos generativos.

## Teoría

Conversión a Markdown de las slides del curso, en [`teoria/`](teoria/):

- [`0 - Programa del módulo.md`](<teoria/0 - Programa del módulo.md>) — las 33 clases en 7 módulos
- [`Configuración del entorno.md`](<teoria/Configuración del entorno.md>) — Clase 2: instalaciones requeridas, con el equivalente para macOS y Apple Silicon

### Material recibido, pendiente de convertir

Las slides de las Clases 3 y 4 ya están en `teoria/` como PDF; falta pasarlas a Markdown.

| Clase | PDF | Contenido |
|:--:|---|---|
| 3 | `Introducción a TensorFlow - Parte 1` | Qué es TensorFlow, tensores, grafo estático, Keras |
| 3 | `Introducción a TensorFlow - Parte 2` | Acceso a datos con `keras.datasets`, modelo `Sequential` y API funcional, `compile` con optimizador, pérdida y métricas |
| 3 | `Introducción a TensorFlow - Parte 3` | Entrenamiento con `model.fit` y evaluación con `model.evaluate` |
| 4 | `Introducción a PyTorch - Parte 1` | Qué es PyTorch, grafo dinámico, GPU, tensores |
| 4 | `Introducción a PyTorch - Parte 2` | `torchvision`, transformaciones (`ToTensor`, `Resize`, `Normalize`) y `DataLoader` |
| 4 | `Introducción a PyTorch - Parte 3` | Modelos con `nn.Module`, funciones de pérdida de `torch.nn` y optimizadores de `torch.optim` |
| 4 | `Introducción a PyTorch - Parte 4` | Bucle de entrenamiento paso a paso (`zero_grad`, `backward`, `step`) y evaluación con `model.eval()` y `torch.no_grad()` |

Entre las dos clases queda cubierto el mismo flujo en los dos frameworks: **acceso a datos → definir el modelo → configurar pérdida y optimizador → entrenar → evaluar**.

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
