# Redes Adversarias Generativas (GANs)

> Conversión a Markdown de las slides del curso (Clase 28 — Redes Adversarias Generativas, dentro del **Módulo 6 — Autoencoders y Modelos Generativos**). El PDF original está en [`material/`](<material/Redes Adversarias Generativas (GANs).pdf>). Continúa de [Autoencoders - P2](<Autoencoders - P2.md>).

## 1. Qué es una GAN

Una **Red Adversaria Generativa** (*Generative Adversarial Network*, GAN) es una arquitectura capaz de **generar datos nuevos** a partir de un conjunto de entrenamiento, construida a partir de dos modelos que se entrenan a la vez:

- **Generador** (`G`): toma un vector de ruido aleatorio, tomado de un **espacio latente de baja dimensión**, y produce datos sintéticos (por ejemplo, una imagen).
- **Discriminador** (`D`): es un clasificador binario que recibe una muestra (real o generada) y decide si es **real** (viene del conjunto de entrenamiento) o **falsa** (la produjo el generador).

El material la llama **antagónica** porque entrena las dos redes **enfrentadas entre sí**, en términos de teoría de juegos: el generador intenta engañar al discriminador, y el discriminador intenta no dejarse engañar. Ambas mejoran durante el entrenamiento, cada una empujada por el progreso de la otra.

> **Nota:** el material no explica **por qué** este enfrentamiento produce, en la práctica, un generador capaz de crear datos realistas. La intuición es que el discriminador actúa como una función de pérdida **aprendida y cambiante**: en vez de decirle al generador "así de lejos estás del dato real" con una métrica fija (como el error cuadrático medio de un autoencoder), el discriminador le da una señal binaria que se vuelve cada vez más exigente a medida que él mismo mejora distinguiendo lo real de lo falso. El generador se ve forzado a producir muestras cada vez más convincentes para seguir "pasando" esa evaluación cada vez más estricta.

## 2. El juego minimax

> **Nota:** toda esta sección es ampliación; el material no incluye ninguna fórmula. Se agrega acá porque es el mecanismo formal detrás del diagrama de la sección 3, y sin él "antagónica" queda como una metáfora sin contenido.

El entrenamiento de una GAN se formula como un juego de suma cero entre `D` y `G`, con la siguiente función de valor:

```
min_G max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

donde `x` es una muestra real del conjunto de entrenamiento, `z` es un vector de ruido tomado del espacio latente, `G(z)` es la muestra generada a partir de ese ruido, y `D(·)` es la probabilidad que asigna el discriminador de que su entrada sea real.

- El **discriminador** quiere **maximizar** `V`: que `D(x)` sea cercano a 1 (acierta con lo real) y que `D(G(z))` sea cercano a 0 (acierta con lo falso).
- El **generador** quiere **minimizar** `V`: quiere que `D(G(z))` sea cercano a 1, es decir, que el discriminador confunda su salida con un dato real.

En el punto teórico de equilibrio de este juego, el generador reproduce la distribución real de los datos tan bien que el discriminador ya no puede distinguir mejor que al azar (`D(x) = 0.5` para cualquier entrada).

### 2.1 Por qué se usa la pérdida no saturante

Con la formulación minimax de arriba, la pérdida que el generador tiene que **minimizar** es `log(1 - D(G(z)))`. Al principio del entrenamiento el generador es malo, produce muestras que el discriminador rechaza con confianza (`D(G(z))` cercano a 0), y en esa región la función `log(1 - D(G(z)))` tiene **gradiente casi plano**: el generador recibe una señal muy débil para mejorar justo cuando más la necesita. Este problema se conoce como **saturación del gradiente**.

En la práctica, en lugar de minimizar `log(1 - D(G(z)))`, se entrena al generador para **maximizar** `log(D(G(z)))` — la llamada **pérdida no saturante**. Es un cambio de objetivo, no una reformulación algebraica exacta del mismo mínimo: el punto de equilibrio teórico del juego no cambia (el generador sigue queriendo que `D(G(z))` se acerque a 1), pero el gradiente respecto a los parámetros del generador es mucho más fuerte cuando `D(G(z))` está cerca de 0, que es exactamente la situación típica del arranque del entrenamiento. Este ajuste es el que se usa en la implementación práctica de casi cualquier GAN, aunque los libros y el material introductorio suelen presentar primero la formulación minimax "pura" de la sección anterior.

## 3. Arquitectura y flujo de entrenamiento

El material ilustra el flujo con dos diagramas equivalentes: uno con notación formal (espacio latente de baja dimensión → red generadora → imágenes falsas; espacio de muestras de alta dimensión → imágenes reales; ambas van a una red discriminadora que devuelve "Real" o "Fake") y otro con un ejemplo de rostros (un conjunto de entrenamiento de caras reales por un lado, ruido que pasa por el generador para producir una cara falsa por el otro; ambas entran al discriminador, que las clasifica como reales o falsas).

El **entrenamiento alternado** — no explicitado como tal en la slide, pero es lo que representan las flechas punteadas de realimentación en el segundo diagrama, que van desde la salida del discriminador de vuelta hacia el generador y hacia el propio discriminador — sigue, en cada iteración, dos pasos separados:

1. **Paso del discriminador**: se le muestra un lote de datos reales y un lote de datos generados por `G` (con los pesos de `G` congelados en este paso). Se actualizan solo los pesos de `D`, para que clasifique mejor reales vs. falsos.
2. **Paso del generador**: se generan nuevas muestras con `G` y se les pide la predicción a `D` (con los pesos de `D` congelados en este paso). Se actualizan solo los pesos de `G`, usando la pérdida no saturante de la sección 2.1, para que las muestras engañen más al discriminador.

> **Nota:** el material presenta el diagrama de bloques y el ejemplo de caras, pero no describe el procedimiento de entrenamiento paso a paso ni menciona que los pesos de una red se congelan mientras se entrena la otra. Esta sección amplía el diagrama con el procedimiento estándar de entrenamiento alternado de una GAN.

## 4. Modos de falla

> **Nota:** esta sección completa entera es ampliación; el material no menciona ninguno de estos problemas, aunque son consecuencia directa de cómo está planteado el juego minimax de la sección 2, y son la razón por la que entrenar una GAN en la práctica es notoriamente más inestable que entrenar una red supervisada convencional.

Al ser un juego entre dos redes que se entrenan simultáneamente (y no la optimización de una única función de pérdida convexa), el entrenamiento de una GAN puede fallar de formas que no tienen equivalente en el entrenamiento supervisado:

- **Mode collapse (colapso de modos)**: el generador descubre que un puñado de muestras (o incluso una sola) engañan sistemáticamente al discriminador, y deja de explorar el resto de la diversidad del conjunto de entrenamiento. El resultado es un generador que produce siempre variaciones de lo mismo, en vez de cubrir toda la distribución real de los datos.
- **Inestabilidad de entrenamiento**: como `G` y `D` se actualizan uno en función del otro, es fácil que entren en una dinámica oscilante donde ninguno converge: el discriminador mejora, el generador se adapta a ese nuevo discriminador, el discriminador vuelve a mejorar contra ese generador, y así sucesivamente sin estabilizarse. Las pérdidas de ambas redes pueden oscilar en vez de bajar de forma monótona, a diferencia de una red entrenada con una única función de pérdida.
- **No convergencia / desequilibrio discriminador-generador**: si el discriminador se vuelve demasiado bueno demasiado rápido, empieza a rechazar con total confianza cualquier muestra generada (`D(G(z))` cercano a 0 de forma sostenida), y ahí reaparece el problema de gradiente casi nulo señalado en 2.1: el generador deja de recibir señal útil para mejorar y el entrenamiento se estanca. El equilibrio buscado —que ambas redes mejoren a un ritmo comparable— es difícil de sostener y no está garantizado por la formulación teórica del juego.

## 5. Aplicaciones

El material cierra con cuatro aplicaciones de las GANs:

1. **Generación de nuevos ejemplos para conjuntos de datos**: generar datos sintéticos adicionales, lo bastante realistas como para complementar el conjunto de entrenamiento original.
2. **Generación de imágenes realistas**: capacidad de producir imágenes que parecen tomadas del mundo real. El material cita **StyleGAN** como ejemplo.
3. **Superresolución de imágenes**: mejorar la calidad de una imagen de baja resolución aumentando su tamaño sin perder detalle, con el modelo **SRGAN**.
4. **Traducción de texto a imagen**: crear imágenes a partir únicamente de descripciones textuales, con el modelo **DALL-E**.

> **Nota:** el material lista estas cuatro aplicaciones como títulos con una oración cada una, sin desarrollarlas. Vale aclarar, como precisión que el material no hace, que DALL-E en sus versiones más conocidas (DALL-E 2 en adelante) es un modelo de **difusión**, no una GAN pura — la cita como ejemplo de "traducción de texto a imagen" es válida como aplicación de esa tarea, pero no como ejemplo arquitectónico estricto de GAN. No se afirma acá ninguna versión o detalle de API de estos modelos por no estar verificado; se deja constancia únicamente de la familia de modelo (difusión vs. adversario) porque es relevante para no confundir el linaje arquitectónico.

## 6. Relación con el resto del módulo

- Esta clase (28) continúa el **Módulo 6 — Autoencoders y Modelos Generativos**, después de [Autoencoders - P1](<Autoencoders - P1.md>) y [Autoencoders - P2](<Autoencoders - P2.md>) (Clase 27). El propio título de la slide de apertura del PDF fuente ("Autoencoders y Modelos Generativos") retoma el título del módulo.
- El autoencoder variacional (VAE), cierre de [Autoencoders - P2](<Autoencoders - P2.md>), y la GAN de esta clase son las dos familias clásicas de modelos generativos que se enseñan antes de los modelos de difusión: ambas aprenden a producir datos nuevos, pero por mecanismos muy distintos — el VAE optimiza una única función de pérdida sobre una distribución latente explícita, mientras que la GAN no tiene una función de pérdida única sino un juego entre dos redes sin distribución latente explícita más que el ruido de entrada del generador.
- El material no lo señala, pero conviene marcarlo: a diferencia del VAE, la GAN **no ofrece una forma directa de inferir el vector latente que corresponde a un dato real dado** (no hay "encoder" en la arquitectura básica de GAN); solo genera datos nuevos hacia adelante, a partir de ruido.
