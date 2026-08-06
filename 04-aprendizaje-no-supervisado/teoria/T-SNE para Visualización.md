# t-SNE para Visualización

> Conversión a Markdown de la slide del curso (Clase 26, Módulo 5). El PDF original está al lado.

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) es una técnica de reducción de dimensionalidad **no lineal** que se utiliza principalmente para la **visualización** de datos de alta dimensión. Es especialmente eficaz para proyectar datos complejos a un espacio de **2D o 3D**, facilitando ver estructuras subyacentes, clusters y patrones.

## 1. ¿Cómo funciona t-SNE?

### Similitud en alta y baja dimensión
- t-SNE calcula la **similitud entre puntos** en el espacio de alta dimensión usando **distribuciones de probabilidad**.
- Luego busca una representación en un **espacio reducido** que **preserve esas similitudes**.

### Minimización de la divergencia de Kullback-Leibler (KL)
- t-SNE minimiza la **divergencia KL** entre la distribución de similitudes en alta dimensión y la de baja dimensión.
- Esto asegura que las **relaciones de proximidad locales** se mantengan (los puntos cercanos en el original quedan cercanos en la proyección).

## 2. Aplicaciones prácticas

- **Visualización de datos de alta dimensionalidad:** datos genómicos, características de imágenes (ej. dígitos MNIST proyectados a 2D, separando cada dígito).
- **Exploración de clusters:** ayuda a identificar y visualizar clusters incluso cuando no son evidentes en el espacio original.
- **Análisis de datos textuales:** visualizar embeddings de palabras o documentos, mostrando relaciones semánticas y temas.
- **Visualización de resultados de modelos de ML:** ver cómo un modelo separa clases en el espacio de características.
- **Detección de anomalías:** identificar outliers observando cómo se dispersan los puntos en el espacio reducido.

## 3. Consideraciones

- Es una técnica **no supervisada** orientada a **visualización**, no a preprocesamiento (no preserva bien las distancias globales ni es determinista).
- **Computacionalmente intensivo** en datasets grandes.
- Preserva bien la **estructura local**, pero la **estructura global** (distancias entre clusters) puede no ser fiable.

> Comparación con PCA (lineal), LDA (supervisado) y UMAP en [Comparación de Técnicas de Reducción de Dimensionalidad](<Comparación de Técnicas de Reducción de Dimensionalidad.md>). Práctica: [notebooks/tsne_practica.ipynb](../notebooks/tsne_practica.ipynb).
