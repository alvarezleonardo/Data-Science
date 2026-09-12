# Capas totalmente conectadas

> Conversión a Markdown de las slides del curso (Clase 8 — Introducción a las CNNs, bloque 06). El PDF original está en [`material/`](material/).

## 1. Qué son y dónde van

Las **capas totalmente conectadas** (también llamadas **capas densas** o *fully connected layers*) se colocan en la **fase final** del modelo. Su función principal es tomar las características que ya extrajeron las capas convolucionales y de agrupamiento, y usarlas para **tomar la decisión final** de clasificación.

El diagrama del curso retoma el pipeline completo, ahora con más detalle en la mitad final:

```
Imagen → Extracción de características → Aplanamiento → Capas totalmente conectadas → SoftMax → Salida
 (cebra)   (convolución + pooling,          (vector)      (capas densas)              (probabilidades)
            varios bloques apilados,
            los mapas se van achicando)
```

En el ejemplo del diagrama entra la foto de una cebra, se aplican sucesivos bloques de extracción de características (cada uno reduce el tamaño espacial y aumenta la cantidad de mapas — se ve en el dibujo cómo los "cubos" se achican pero se apilan más), y el resultado se aplana en un vector que alimenta una red densa de varias capas. La salida final, después de la función de activación **SoftMax**, da una probabilidad por clase: caballo 0,2 · **cebra 0,7** · perro 0,1 — el mismo ejemplo que aparece en la introducción del módulo.

## 2. Por qué hace falta aplanar

Las capas convolucionales trabajan con **volúmenes** (alto × ancho × canales), pero una capa densa espera un **vector** de una sola dimensión: cada neurona de una capa `Linear`/densa se conecta con cada valor de la entrada, y esa conexión solo tiene sentido sobre una lista plana de números, no sobre una grilla.

El *flattening* toma el último mapa de características (por ejemplo, de forma `128 × 4 × 4` en el notebook del curso) y lo convierte en un vector de `128 × 4 × 4 = 2048` valores, sin cambiar ni perder ningún dato — solo reordena la memoria en una fila.

## 3. Por qué antes se usa max pooling y no average pooling

Antes del aplanamiento, cada bloque convolucional suele reducir el tamaño espacial con una capa de **pooling**. La variante más común es el **max pooling**, que se queda con el valor máximo de cada región (por ejemplo, cada ventana de 2×2), en vez del **average pooling**, que promedia los valores de la región.

La diferencia no es solo aritmética: importa qué representa cada valor del mapa de características. Después de la convolución y la ReLU (ver [`Capa convolucional - P2.md`](Capa%20convolucional%20-%20P2.md)), un valor alto en el mapa indica que el filtro **encontró fuertemente** el patrón que detecta (un borde, una textura) en esa posición. El **máximo** conserva esa activación más fuerte —la evidencia más clara de que el rasgo está presente—, mientras que el **promedio** la diluye mezclándola con posiciones vecinas donde el patrón puede no estar tan presente, o directamente ausente. Por eso max pooling tiende a preservar mejor la información relevante para la clasificación, y es la opción por defecto en la mayoría de las arquitecturas de CNN, incluida la del notebook del curso (`MaxPool2d(2, 2)`).

## 4. Cuántos parámetros tiene una capa densa, y por qué domina el conteo

A diferencia de una capa convolucional —donde un mismo filtro chico se reutiliza en toda la imagen (ver [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md), sección 4)—, en una capa densa **cada neurona de salida tiene un peso distinto para cada valor de entrada**, más un sesgo:

```
parámetros de una capa densa = (entradas × salidas) + salidas (sesgos)
```

Aplicado a la red del notebook (`CNN_pytorch.ipynb`), que después del aplanamiento tiene las capas densas 2048 → 512 → 256 → 10:

| Capa | Cálculo | Parámetros |
|---|---|---|
| Densa 1 (2048 → 512) | 2048 × 512 + 512 | **1.049.088** |
| Densa 2 (512 → 256) | 512 × 256 + 256 | 131.328 |
| Densa 3 (256 → 10) | 256 × 10 + 10 | 2.570 |

De un total de **1.276.234 parámetros** en toda la red, las tres capas convolucionales (`Conv2d(3→32→64→128)`) suman apenas **93.248** (7%), mientras que la **primera capa densa sola** concentra **1.049.088** (82%). La razón es exactamente la que explica esta sección: la convolución reutiliza el mismo filtro chico en toda la imagen, mientras que la primera capa densa conecta cada uno de los 2048 valores del vector aplanado con cada una de las 512 neuronas siguientes, sin ningún tipo de reuso de pesos. Aplanar demasiado tarde (con mapas todavía grandes) es, en la práctica, la causa más común de que una CNN tenga muchísimos más parámetros de los necesarios.

## 5. La salida: SoftMax

La última capa de la red aplica la función de activación **SoftMax**, que convierte los valores de salida (uno por clase) en una distribución de probabilidad: todos los valores quedan entre 0 y 1, y suman 1 entre todas las clases. Es la misma función de activación que cierra el MLP del [Módulo 06](../../06-fundamentos-de-redes-neuronales/teoria/), y ya se había mencionado en [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](01%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%201.md).

> **Nota:** si el modelo se entrena con `nn.CrossEntropyLoss` en PyTorch (como en los notebooks del curso), la última capa densa **no** debe aplicar SoftMax explícitamente — `CrossEntropyLoss` ya la aplica internamente. Este mismo punto se explica en detalle en [`Introducción a PyTorch - Parte 3.md`](../../07-fundamentos-de-deep-learning/teoria/Introducci%C3%B3n%20a%20PyTorch%20-%20Parte%203.md#2-un-mlp-concreto).

## 6. Aplicaciones de las CNN

La última slide del deck lista tres aplicaciones típicas de las CNN, más allá de la clasificación simple usada como ejemplo en todo el módulo:

| Aplicación | Qué resuelve |
|---|---|
| **Clasificación de imágenes** | asignar una única etiqueta a toda la imagen (el caso desarrollado en este módulo: caballo / cebra / perro) |
| **Detección de objetos** | ubicar y clasificar múltiples objetos dentro de una misma imagen, con su posición (recuadros delimitadores) |
| **Reconocimiento facial** | identificar o verificar la identidad de una persona a partir de rasgos faciales |

## 7. Relación con el resto del módulo

- Cierra el recorrido de la Clase 8: [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](01%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%201.md) presentó la arquitectura completa; [`02 - Introducción a las redes neuronales convolucionales - Parte 2.md`](02%20-%20Introducci%C3%B3n%20a%20las%20redes%20neuronales%20convolucionales%20-%20Parte%202.md) la inspiración biológica y la imagen como matriz de píxeles; [`Capa convolucional - P1.md`](Capa%20convolucional%20-%20P1.md) y [`Capa convolucional - P2.md`](Capa%20convolucional%20-%20P2.md) la mecánica de filtros, padding, stride y ReLU; este archivo cierra con la mitad "que decide" de la red.
- Todo lo descrito acá está implementado en `../notebooks/CNN_pytorch.ipynb`: los tres bloques `Conv2d + MaxPool2d` (extracción de características), el `Flatten` (aplanamiento) y las tres capas `Linear` (2048→512→256→10) con SoftMax implícito en `CrossEntropyLoss`.
- La segunda implementación del mismo modelo, en TensorFlow/Keras, está en `../notebooks/OD_RN2_ESP_M03_S13_CNNs_en_TensorFlow_Recurso_descargable.ipynb`, con las mismas capas expresadas como `Conv2D`, `MaxPooling2D`, `Flatten` y `Dense`.
- El MLP y la función SoftMax ya se habían visto, para el caso general de redes densas, en el [Módulo 06 — Fundamentos de redes neuronales](../../06-fundamentos-de-redes-neuronales/teoria/).
