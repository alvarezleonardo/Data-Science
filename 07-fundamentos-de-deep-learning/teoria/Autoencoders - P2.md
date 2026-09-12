# Autoencoders - P2

> Conversión a Markdown de las slides del curso (Clase 27 — Autoencoders y Modelos Generativos). El PDF original está en [`material/`](<material/Autoencoders - P2.pdf>). Continúa de [Autoencoders - P1](<Autoencoders - P1.md>).

## 1. Tipos de autoencoders

El material presenta cuatro variantes de la arquitectura básica vista en [Autoencoders - P1](<Autoencoders - P1.md>), cada una pensada para un objetivo distinto sobre el espacio latente o sobre la robustez del modelo:

| Tipo | Qué hace | Para qué sirve |
|---|---|---|
| **Autoencoders dispersos** (*sparse*) | Reducen el número de neuronas que pueden activarse al mismo tiempo. Para imponer esta restricción de dispersión se suele usar la **regularización L1**. | Fuerza a que solo unas pocas neuronas del espacio latente estén "activas" para cada entrada, lo que favorece representaciones más interpretables y evita que el autoencoder aprenda una simple identidad. |
| **Autoencoders contractivos** | Logran una representación latente más estable y robusta, de modo que pequeños cambios o perturbaciones en los datos de entrada no afecten drásticamente la representación interna del modelo. | Mejora la robustez ante ruido pequeño o variaciones menores en la entrada. |
| **Autoencoders de eliminación de ruido** (*denoising*) | Reciben datos de entrada parcialmente corruptos y se entrenan para restaurar la entrada original, eliminando la información inútil. Evitan el sobreajuste y son útiles para limpiar archivos de imagen y audio ruidosos o dañados. | Tareas de limpieza de ruido (*denoising*) sobre imágenes o audio. |
| **Autoencoders variacionales** (VAE) | Aprenden una **distribución probabilística** de los datos de entrada, en lugar de un único punto fijo en el espacio latente. Esto les permite generar nuevas muestras a partir de esa distribución. | Es la puerta de entrada a los **modelos generativos**: en vez de solo reconstruir una entrada dada, el modelo puede muestrear puntos nuevos del espacio latente aprendido y generar datos que nunca vio. |

> **Nota:** el material enumera estas cuatro variantes en paralelo, como si fueran alternativas equivalentes, pero conviene distinguir dos grupos con objetivos distintos. Los autoencoders **dispersos**, **contractivos** y de **eliminación de ruido** son básicamente regularizaciones sobre el autoencoder clásico de [Autoencoders - P1](<Autoencoders - P1.md>): buscan una representación latente mejor o más robusta, pero siguen mapeando cada entrada a un punto (o vector) fijo del espacio latente. El autoencoder **variacional**, en cambio, cambia la naturaleza del espacio latente: en vez de un vector fijo, aprende **parámetros de una distribución** (típicamente media y varianza de una normal) para cada entrada, y es ese cambio el que habilita la generación de datos nuevos. Es, de las cuatro variantes, la única que es estrictamente un **modelo generativo** en el sentido de poder producir muestras nuevas y no solo reconstruir o limpiar una entrada dada.

> **Nota:** el material no explica el mecanismo interno de ninguna de las cuatro variantes (no hay fórmula de la regularización L1 aplicada a la dispersión, ni del término de penalización por contracción, ni del proceso de corrupción de la entrada en el denoising, ni de la reparametrización que usan los VAE para poder entrenarse con descenso de gradiente pese a la naturaleza probabilística del muestreo). Es un listado de alto nivel; el detalle matemático de cada variante queda fuera del alcance de esta clase según el propio material.

## 2. Aplicaciones

El material cierra con cinco aplicaciones prácticas de los autoencoders:

1. **Compresión de datos**: la representación en el espacio latente ocupa menos espacio que la entrada original.
2. **Reducción de dimensionalidad**: análogo no lineal a técnicas como PCA, útil para visualizar o preprocesar datos de alta dimensión.
3. **Detección de anomalías y reconocimiento facial**: si el autoencoder se entrena solo con datos "normales", una entrada anómala se reconstruye peor (mayor error de reconstrucción), lo que sirve como señal de anomalía; en reconocimiento facial, el espacio latente sirve como representación compacta de una cara para comparar identidades.
4. **Eliminación de ruido en imágenes y audio**: la aplicación directa de los autoencoders de eliminación de ruido de la sección 1.
5. **Generación de datos**: la aplicación directa de los autoencoders variacionales — generar imágenes, audio u otro tipo de dato nuevo a partir de puntos muestreados del espacio latente aprendido.

> **Nota:** el material lista estas cinco aplicaciones sin desarrollarlas (son títulos de un slide, sin texto adicional). La explicación de cómo se relaciona cada una con el tipo de autoencoder correspondiente (por ejemplo, que la detección de anomalías se apoya en el error de reconstrucción, o que la generación de datos depende puntualmente del autoencoder variacional) es una ampliación de estas notas y no está en la slide original.

## 3. Relación con el resto del módulo

- Esta clase completa, junto con [Autoencoders - P1](<Autoencoders - P1.md>), la introducción a los autoencoders dentro del **Módulo 6 — Autoencoders y Modelos Generativos** (clases 27 a 31).
- El autoencoder variacional (VAE), introducido acá como una variante más, es probablemente el punto de partida de las clases siguientes del módulo dedicadas específicamente a **modelos generativos** — el título del módulo ya anticipa que el contenido no se agota en el autoencoder clásico de compresión/reconstrucción.
- Las aplicaciones de detección de anomalías y reducción de dimensionalidad conectan con contenido de módulos anteriores del programa (aprendizaje no supervisado, PCA) vistos fuera de este módulo de deep learning, aunque el material no hace esa conexión explícita.
