# Aplicaciones de PCA

> Conversión a Markdown de la slide del curso (Clase 24, Módulo 5). El PDF original está al lado.

## 1. Visualización de datos de alta dimensionalidad

PCA es muy útil para **visualizar** datos de muchas dimensiones en un espacio reducido. Permite reducir a **2 o 3 componentes principales**, capturando la mayor parte de la variabilidad, lo que facilita ver **patrones y clusters** que en el espacio original serían imposibles de graficar.

> Ejemplo: un dataset con muchas columnas se proyecta a 2 componentes principales y se colorea por etiqueta de cluster → los grupos se distinguen claramente en 2D.

## 2. Aplicaciones en Machine Learning — Preprocesamiento y reducción del overfitting

PCA puede usarse para **reducir la dimensionalidad antes de entrenar** un modelo de ML, mejorando el rendimiento y **evitando el overfitting**. Además, reducir dimensiones **mejora la velocidad y la escalabilidad** de los algoritmos.

- Menos dimensiones → menos ruido → mejor **precisión** en algunos casos.
- Menos features → entrenamiento más **rápido**.
- **Cuidado:** al ser preprocesamiento, PCA se ajusta **solo con el train** (`fit` en train, `transform` en train y test) para evitar fuga de información.

## 3. Ejemplos de aplicaciones

- **Reconocimiento de imágenes:** reducir la dimensionalidad de las imágenes antes de aplicar algoritmos de clasificación → proceso más eficiente (ej. dígitos MNIST proyectados a 2 componentes, separando los dígitos).
- **Procesamiento de texto:** reducir la dimensionalidad de representaciones de texto (matrices **términos-documentos**), facilitando el análisis y la clasificación de documentos.

## Resumen de casos de uso

| Caso | Para qué sirve PCA |
|------|--------------------|
| **Visualización** | Proyectar a 2-3D para ver clusters/tendencias |
| **Preprocesamiento ML** | Reducir ruido y overfitting, acelerar entrenamiento |
| **Imágenes** | Comprimir la representación antes de clasificar |
| **Texto (NLP clásico)** | Reducir matrices términos-documentos dispersas |

> Enlaza con [Análisis de Componentes Principales (PCA)](<Análisis de Componentes Principales (PCA).md>) (teoría, Clase 23). Práctica: [notebooks/aplicaciones_pca_practica.ipynb](../notebooks/aplicaciones_pca_practica.ipynb).
