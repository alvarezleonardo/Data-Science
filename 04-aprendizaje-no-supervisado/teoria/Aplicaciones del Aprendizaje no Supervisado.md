# Aplicaciones del Aprendizaje no Supervisado

> Conversión a Markdown de la slide del curso (Clase 4). El PDF original está al lado.

## 1. Importancia del aprendizaje no supervisado en la IA

El **aprendizaje no supervisado** no puede aplicarse directamente a problemas de **clasificación** o **regresión**, ya que, a diferencia del aprendizaje supervisado, aquí se dispone de los **datos de entrada pero no de los datos de salida**. Su objetivo principal es:

- Descubrir la **estructura oculta** en el conjunto de datos.
- **Agrupar** los datos según sus similitudes.
- Proporcionar una **representación útil** del conjunto.

Es una técnica para analizar datos **sin procesar** y transformarlos en información útil. Se compara con el **aprendizaje humano** (se acerca más a una verdadera IA), porque el modelo aprende a identificar patrones **sin supervisión**.

Trabaja con datos **no etiquetados ni categorizados**, lo que lo hace especialmente relevante en contextos donde no se cuenta con datos de salida o etiquetas predeterminadas (las situaciones con datos de entrada y salida completos son limitadas).

## 2. Algoritmos de aprendizaje no supervisado

Tres enfoques centrales:

### Agrupamiento (clustering)

Método que organiza los objetos de modo que aquellos que comparten muchas **similitudes** queden en un mismo grupo. Este análisis, conocido como **análisis de conglomerados**, identifica los puntos en común entre los objetos y los clasifica según la presencia o ausencia de esas similitudes.

### Asociación

Una **regla de asociación** es un método usado para descubrir **relaciones entre las variables** dentro de grandes bases de datos. Muy útil en **marketing**: por ejemplo, identificar que los clientes que compran el artículo X (un teléfono) suelen comprar también el artículo Y (un cargador o auriculares) → *market basket analysis*.

### Algoritmos populares

- **KNN** (K-Nearest Neighbors, vecinos más cercanos)
- **Agrupación jerárquica**
- **Detección de anomalías**
- **Redes neuronales**
- **PCA** (Análisis de Componentes Principales)
- **ICA** (Análisis de Componentes Independientes)
- **Algoritmo a priori**
- **SVD** (Descomposición en Valores Singulares)

### Ventajas
- Adecuado para tareas **más complejas** que el supervisado, porque **no requiere datos etiquetados**.
- Es **más fácil obtener datos no etiquetados** que etiquetados → útil con grandes volúmenes de datos.

### Desventajas
- Es **más complejo** que el supervisado, ya que no cuenta con resultados conocidos ni etiquetas previas.
- El resultado puede ser **menos preciso**, porque los datos de entrada no están etiquetados y el algoritmo no conoce la salida exacta de antemano.

## 3. Casos de uso

- **Segmentación de clientes:** agrupar clientes por comportamiento y características, sin etiquetas predefinidas, para diseñar estrategias de marketing dirigidas.
- **Detección de anomalías:** identificar patrones inusuales (p. ej. fraude en transacciones financieras) sin ejemplos previos de fraude.
- **Reducción de dimensionalidad:** técnicas como **PCA** y **t-SNE** reducen la cantidad de variables, facilitando la visualización e interpretación de datos complejos.
- **Agrupamiento de documentos:** en NLP, agrupar documentos similares en clusters para organizar grandes colecciones de texto y recomendar artículos/noticias.
- **Compresión de imágenes:** algoritmos como el **autoencoder** comprimen imágenes reduciendo su tamaño sin perder información significativa.
- **Análisis de redes sociales:** identificar comunidades y relaciones agrupando usuarios con comportamientos o intereses similares.
- **Descubrimiento de estructura en datos:** revelar estructuras ocultas y patrones no evidentes en grandes conjuntos.
- **Sistemas de recomendación:** encontrar patrones en los datos de usuarios (vía clustering) para recomendaciones más precisas.
- **Modelado de temas:** algoritmos como **LDA** (Latent Dirichlet Allocation) descubren temas subyacentes en un conjunto de documentos.
- **Reconocimiento de patrones:** en visión por computadora, identificar patrones en imágenes y videos (detección de objetos, segmentación) sin etiquetado exhaustivo.
