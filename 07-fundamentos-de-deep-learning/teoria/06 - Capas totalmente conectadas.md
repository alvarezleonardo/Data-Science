# Capas totalmente conectadas

> Conversión a Markdown de las slides del curso (Clase 11 — Capas totalmente conectadas, bloque 06). El PDF original está al lado. Cierra el recorrido de la arquitectura que empezó en [`01 - Introducción a las redes neuronales convolucionales - Parte 1.md`](<01 - Introducción a las redes neuronales convolucionales - Parte 1.md>).

## 1. Qué son y dónde van

Las **capas totalmente conectadas** —también llamadas **capas densas** o *fully connected layers*— se colocan en la **fase final del modelo**. Su función principal es **usar las características extraídas por las capas convolucionales y de agrupamiento para tomar la decisión final**.

El diagrama de la slide es el mismo de la Clase 8, ahora leído desde el final:

```
Extracción de características  →  Capas totalmente conectadas  →  SoftMax  →  Salida
   (convolución + pooling)              (densas)                         Caballo / Cebra / Perro
```

La división de trabajo es la que se viene repitiendo en todo el módulo: la primera mitad de la red **aprende qué mirar**, la segunda **decide**.

## 2. Qué significa "totalmente conectada"

Cada neurona de la capa se conecta con **todas** las salidas de la capa anterior. Es la capa clásica del MLP del módulo 06 (cap. 30 del manual): una multiplicación matriz-vector más un sesgo, seguida de una activación.

La diferencia con la parte convolucional está justamente ahí:

| | Capa convolucional | Capa densa |
|---|---|---|
| Conexión | local: cada salida mira una ventana chica | global: cada neurona ve todo |
| Pesos | compartidos entre posiciones | uno por conexión |
| Conserva la posición | sí, la salida sigue siendo un mapa | no, la entrada es un vector plano |
| Parámetros | pocos | muchos |

Entre las dos mitades hace falta un paso de traducción: el **aplanamiento** (*flattening*) de la Clase 10, que convierte los mapas de características en un vector. En los notebooks del módulo, tres bloques de convolución y pooling dejan 128 mapas de 4×4, que aplanados dan un vector de **2.048** valores.

Ahí se concentra el costo: una densa de 2.048 → 512 son 2.048 × 512 + 512 = **1.049.088 parámetros**, más que toda la parte convolucional junta. Por eso se pone el pooling antes: no solo resume, además hace que esta capa sea manejable.

## 3. La salida y softmax

La última capa densa tiene **una neurona por clase**, y sobre ella se aplica la función de activación **SoftMax**.

Softmax convierte los valores crudos (*logits*) en una distribución de probabilidad: todos entre 0 y 1, y sumando 1. En el ejemplo de la slide: caballo, cebra, perro. La predicción es la clase con el valor más alto.

Va con la pérdida de **entropía cruzada categórica** (cap. 34 del manual). Un detalle de implementación que aparece en los dos notebooks del módulo:

- En **Keras** se declara `activation='softmax'` en la última capa y se compila con `categorical_crossentropy`.
- En **PyTorch** la última capa **no lleva softmax**: `nn.CrossEntropyLoss` ya lo aplica internamente sobre los logits. Agregarlo a mano lo aplicaría dos veces y empeoraría el entrenamiento.

## 4. Aplicaciones de las CNN

La slide de cierre nombra tres:

| Aplicación | Qué devuelve |
|---|---|
| **Clasificación de imágenes** | una etiqueta para toda la imagen |
| **Detección de objetos** | qué objetos hay y **dónde**, con su recuadro |
| **Reconocimiento facial** | identidad, comparando representaciones de rostros |

Las tres comparten la primera mitad de la red —la parte convolucional que extrae características— y cambian la segunda. Es la base del *transfer learning*: se reutiliza el extractor ya entrenado y se reemplaza la cabeza densa por la que corresponde a la tarea nueva.

Lo que hacen los notebooks de las clases 12 y 13 es el primer caso: clasificación de imágenes sobre CIFAR-10.

## 5. La arquitectura completa

Con esta clase queda armado el recorrido entero:

```
imagen → [convolución + ReLU → pooling] × N → flatten → densas → softmax → probabilidades
         └────── extracción de características ─────┘   └──── decisión ────┘
```

| Pieza | Clase | Documento |
|---|:---:|---|
| Arquitectura general | 8 | [`01`](<01 - Introducción a las redes neuronales convolucionales - Parte 1.md>) |
| Corteza visual, imagen y canales | 8 | [`02`](<02 - Introducción a las redes neuronales convolucionales - Parte 2.md>) |
| Filtros y convolución | 9 | [`03`](<03 - Capa convolucional - Parte 1.md>) |
| Padding, stride y ReLU | 9 | [`04`](<04 - Capa convolucional - Parte 2.md>) |
| Pooling y aplanamiento | 10 | pendiente — falta el PDF (bloque 05) |
| Capas densas y softmax | 11 | este documento |
| Implementación en PyTorch | 12 | [`cnn_cifar10_pytorch.ipynb`](../notebooks/cnn_cifar10_pytorch.ipynb) |
| Implementación en TensorFlow | 13 | [`OD_RN2_ESP_M03_S13...ipynb`](<../notebooks/OD_RN2_ESP_M03_S13_CNNs_en_TensorFlow_Recurso_descargable.ipynb>) |
