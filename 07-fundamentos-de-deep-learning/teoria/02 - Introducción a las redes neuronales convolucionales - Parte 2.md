# Introducción a las redes neuronales convolucionales — Parte 2

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 02). El PDF original está al lado. Continúa la [Parte 1](<01 - Introducción a las redes neuronales convolucionales - Parte 1.md>), que presentó la arquitectura completa.

## 1. Las CNN y la corteza visual humana

La Parte 1 dijo que las CNN "imitan al ojo humano". Acá el material desarrolla la analogía.

Las CNN se inspiran directamente en cómo la **corteza visual** procesa la información visual. El punto de contacto es el **procesamiento jerárquico**: los dos sistemas arrancan por características simples y las van combinando en representaciones cada vez más complejas.

| En la CNN | En el sistema visual |
|---|---|
| primeras capas convolucionales | detectan características básicas: bordes, colores, cambios de intensidad |
| capas intermedias | combinan esos bordes en texturas, esquinas y formas |
| capas finales | representan objetos completos |

El diagrama de la slide recorre la vía visual: **retina → NGL → V1 → campos receptivos**. El *núcleo geniculado lateral* (NGL) es la estación del tálamo que recibe la señal de la retina y la manda a **V1**, la primera área de la corteza visual. Ahí cada neurona responde solo a una región acotada del campo visual: su **campo receptivo**.

Esa idea de campo receptivo es la que traduce directamente a la red: un filtro convolucional también mira una ventana chica de la imagen por vez, no la imagen entera.

El material cierra con una propiedad que se gana de regalo: las CNN **son invariantes a transformaciones**. Como el mismo filtro se aplica en todas las posiciones, el patrón se detecta esté donde esté. Conviene ser preciso con el alcance: la convolución da invarianza a **traslación**; frente a rotaciones o cambios de escala la red no es invariante por construcción, hay que enseñárselo con datos (*data augmentation*).

## 2. Qué es una imagen

Para que la convolución tenga sentido hay que ver la imagen como lo que es para la red: **números**.

Una imagen es una **matriz bidimensional de valores numéricos**. Cada valor es un **píxel**, la unidad más chica de una imagen digital, y representa el color o la **intensidad de luz** en un punto específico.

La cantidad de píxeles que componen la imagen es su **resolución**. Una imagen de 1920×1080 tiene 2.073.600 píxeles; una de CIFAR-10, apenas 32×32 = 1.024.

En escala de grises alcanza con un número por píxel, típicamente un entero de 0 a 255: 0 es negro, 255 es blanco.

## 3. Canales de color

Las imágenes a color se componen de varios **canales**. Los más comunes son los canales **RGB** (rojo, verde y azul). En una imagen RGB cada píxel se define por una **combinación de tres valores**, uno por canal.

Así que una imagen a color no es una matriz sino un **tensor de tres dimensiones**: `alto × ancho × canales`.

| Imagen | Forma | Valores |
|---|---|---|
| escala de grises 28×28 (MNIST) | 28 × 28 × 1 | 784 |
| color 32×32 (CIFAR-10) | 32 × 32 × 3 | 3.072 |
| color 224×224 (ImageNet) | 224 × 224 × 3 | 150.528 |

Dos consecuencias que se usan todo el tiempo más adelante:

- **El filtro también tiene profundidad.** Si la entrada tiene 3 canales, un filtro de 3×3 es en realidad de 3×3×3: recorre los tres canales a la vez y produce **un solo** mapa de características. Está desarrollado en [`03 - Capa convolucional - Parte 1.md`](<03 - Capa convolucional - Parte 1.md>).
- **El orden de los ejes cambia según el framework.** PyTorch usa `(N, C, H, W)` —canales primero— y TensorFlow/Keras usa `(N, H, W, C)` —canales último—. Es la fuente número uno de errores de shape al pasar código de uno al otro.

Sobre el rango de valores: los píxeles llegan como enteros de 0 a 255 y a la red se le entregan escalados. Los dos notebooks del módulo lo hacen distinto —Keras divide por 255 y deja `[0, 1]`, PyTorch normaliza y deja `[-1, 1]`— y las dos formas son válidas; lo importante es que la escala sea chica y consistente entre entrenamiento y evaluación.

## 4. Qué sigue

Con la imagen entendida como tensor numérico, la Clase 9 define la operación que se le aplica: la **capa convolucional**, con sus filtros, su padding, su stride y su función de activación.
