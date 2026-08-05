# Análisis Discriminante Lineal (LDA)

> Conversión a Markdown de la slide del curso (Clase 25, Módulo 5). El PDF original está al lado.

El **Análisis Discriminante Lineal (LDA)** es una técnica de **reducción de dimensionalidad** y **clasificación supervisada** utilizada para la **separación de clases** en un conjunto de datos. A diferencia de PCA, LDA busca la proyección que **maximiza la separación entre clases** (no la varianza total).

## 1. Objetivos de LDA

LDA busca una **proyección lineal** de los datos que **maximice la separación entre diferentes clases**, encontrando las direcciones en las que los datos de distintas clases están mejor separados. Al igual que PCA reduce dimensiones, pero:
- **PCA:** busca la mayor **varianza** (no supervisado, ignora las etiquetas).
- **LDA:** busca la mayor **separación entre clases** (supervisado, usa las etiquetas).

## 2. Pasos de LDA

1. **Cálculo de las medias:** la media de cada clase y la media global de todas las clases.
2. **Matrices de varianza-covarianza:**
   - **Dentro de la clase** (within-class): varianza-covarianza dentro de cada clase.
   - **Entre clases** (between-class): varianza-covarianza entre las clases.
3. **Cálculo de las proyecciones:** encontrar las direcciones (vectores propios) que **maximizan la razón** entre la varianza *entre clases* y la varianza *dentro de las clases*. Estos vectores forman las nuevas dimensiones (**componentes discriminantes**).
4. **Proyección de datos:** proyectar los datos originales al espacio reducido usando esas direcciones, facilitando la separación entre clases.

> Nota: LDA genera como máximo **(nº de clases − 1)** componentes discriminantes.

## 3. Diferencias entre LDA y PCA

| | **PCA** | **LDA** |
|--|---------|---------|
| **Supervisión** | No supervisado (ignora etiquetas) | **Supervisado** (usa etiquetas) |
| **Qué maximiza** | La **varianza total** de los datos | La **separación entre clases** |
| **Cuándo usar** | Reducir dimensión capturando la mayor varianza, sin considerar estructura de clase | Reducir dimensión manteniendo/mejorando la separación entre clases |
| **Nº de componentes** | Hasta el nº de features | Hasta **(nº de clases − 1)** |

## 4. Aplicaciones de LDA

- **Clasificación:** clasificar datos en categorías según las características proyectadas en el espacio reducido (reconocimiento de patrones, clasificación de textos, diagnóstico médico).
- **Reducción de dimensionalidad en preprocesamiento:** mejora el rendimiento y la interpretabilidad de los modelos de clasificación cuando hay muchas características.
- **Análisis de datos multiclase:** maneja problemas de clasificación **multiclase**, no solo binarios (ej. dataset `wine` con 3 clases).
- **Reconocimiento facial y de imágenes:** mejora la precisión reduciendo la dimensionalidad de las imágenes mientras maximiza la separación entre identidades (ej. dataset de rostros Olivetti).

> Enlaza con [Análisis de Componentes Principales (PCA)](<Análisis de Componentes Principales (PCA).md>) (Clase 23) y los [Principios de Reducción de la Dimensionalidad](<Principios de Reducción de la Dimensionalidad.md>) (Clase 22).
