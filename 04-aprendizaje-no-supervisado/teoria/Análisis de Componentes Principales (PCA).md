# Análisis de Componentes Principales (PCA)

> Conversión a Markdown de la slide del curso (Clase 23, Módulo 5). El PDF original está al lado.

El **Análisis de Componentes Principales (PCA)** es una técnica estadística que **reduce la dimensionalidad** de los datos manteniendo la **mayor cantidad de variabilidad** del conjunto original. Transforma los datos a un **nuevo sistema de coordenadas** donde las nuevas variables (**componentes principales**) capturan la mayor varianza posible.

## 1. Objetivo de PCA

Identificar las **direcciones (componentes principales)** en las que los datos **varían más**. Estas direcciones representan la mayor parte de la variabilidad presente en el conjunto original. En otras palabras, PCA busca una **nueva base** para el espacio de datos donde las nuevas dimensiones son las **direcciones de máxima varianza**.

- Es una técnica **lineal** y **no supervisada** (no usa etiquetas de clase).
- Los componentes son **ortogonales** entre sí (cada uno captura una dirección distinta de varianza).

## 2. Pasos fundamentales de PCA

1. **Centrar los datos:** restar la media de cada variable → `X_centrado = X - X̄`. Asegura que los componentes se calculen con datos centrados en el origen.
2. **Calcular la matriz de covarianza:** captura las varianzas y covarianzas entre variables sobre los datos centrados → `C = (1 / (n-1)) · X_centradoᵀ · X_centrado`.
3. **Calcular valores y vectores propios (eigen):** resolviendo `C · v_i = λ_i · v_i`. Los **vectores propios** son las direcciones principales de dispersión; los **valores propios** indican cuánta varianza explica cada dirección.
4. **Ordenar los componentes:** ordenar los vectores propios por su valor propio de **mayor a menor**. Los primeros capturan la mayor parte de la varianza.
5. **Proyectar los datos:** multiplicar los datos centrados por los vectores propios seleccionados → `X_reducido = X_centrado · W` (donde `W` son los vectores propios elegidos).

## 3. Interpretación de resultados

- **Componentes principales:** son **combinaciones lineales** de las variables originales. Cada componente es **ortogonal** a los demás → captura una dirección distinta de la varianza.
- **Varianza explicada:** la proporción de la varianza total que explica cada componente sirve para decidir **cuántos componentes retener**. El valor propio asociado a cada componente indica la cantidad de varianza que explica.
- **Scree plot (gráfico de codo):** grafica los valores propios (o la varianza explicada) según su orden. Ayuda a elegir el **número óptimo de componentes**, buscando el punto donde la pendiente se **estabiliza** (el "codo").

## 4. Cuándo usar PCA

- Reducir dimensionalidad **conservando información** (varianza).
- **Visualizar** datos de alta dimensión en 2-3D.
- **Preprocesar** antes de entrenar (menos ruido, menos overfitting, más velocidad).
- **Descorrelacionar** features (los componentes son ortogonales).

> **Importante:** conviene **estandarizar** las variables antes de PCA cuando están en escalas distintas, porque PCA es sensible a la escala (una variable con valores grandes dominaría la varianza). Práctica: [notebooks/aplicaciones_pca_practica.ipynb](../notebooks/aplicaciones_pca_practica.ipynb). Comparación con LDA en [Análisis Discriminante Lineal (LDA)](<Análisis Discriminante Lineal (LDA).md>).
