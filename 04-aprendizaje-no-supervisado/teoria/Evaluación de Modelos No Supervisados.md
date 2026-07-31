# Evaluación de Modelos No Supervisados

> Conversión a Markdown de la slide del curso (Clase 7). El PDF original está al lado.

## 1. Introducción a las métricas de evaluación

El aprendizaje no supervisado identifica patrones y estructuras ocultas en datos **no etiquetados**. A diferencia del supervisado, **no existen etiquetas predefinidas** contra las cuales medir; se usan técnicas que exploran y revelan la **estructura intrínseca** de los datos.

Las métricas permiten evaluar:
- La **cohesión** y **separación** de los clusters.
- La **optimización del número de clusters**.
- La capacidad de representar los datos en espacios de **menor dimensión**.

## 2. Índice de Silueta (Silhouette)

Mide la **calidad de la agrupación**. Se calcula **para cada punto** y refleja cuán bien está agrupado respecto a los puntos de su propio grupo y a los del grupo más cercano.

- Varía **entre -1 y 1**.
- Cercano a **1** → el punto está **bien agrupado**.
- Cercano a **-1** → el punto podría estar **mal agrupado**.

Útil para evaluar cohesión y separación en técnicas como K-means.

## 3. Coeficiente de Davies-Bouldin

Evalúa la calidad de los clusters en función de la **distancia promedio intra-cluster** y la **distancia entre clusters**. Se calcula como la media de las razones de dispersión intra-cluster a distancia entre clusters.

- **Valores más bajos = mejor** calidad (clusters más compactos y mejor separados).
- Útil para comparar agrupaciones (K-means, jerárquicos).

## 4. Método del Codo (Elbow Method)

Determina el **número óptimo de clusters**. Grafica la varianza explicada (o inercia) en función del número de clusters y busca el **"codo"**: el punto donde agregar más clusters aporta una **mejora marginal**.

- El "codo" es **subjetivo**: es donde la tasa de disminución de la varianza se estabiliza.
- Se aplica sobre todo con K-means.

## 5. Varianza Explicada (PCA)

En PCA, la **varianza explicada** es la proporción de la varianza total capturada por cada **componente principal**. Ayuda a decidir **cuántos componentes retener**.

- Se expresa como proporción o **porcentaje**; valores altos → el componente explica más varianza.
- Clave para la reducción de dimensionalidad.

> Cada métrica da una perspectiva distinta; combinarlas permite una evaluación más completa.

## 6. Validación cruzada (Cross Validation)

Método para evaluar la **capacidad predictiva** de un modelo asegurando que **generalice** bien a datos no vistos. La idea: en lugar de usar sets separados para entrenar y validar, se usan **todos los datos** rotando el rol.

**Proceso (k-Fold):**
1. **División del dataset** en `k` subconjuntos ("folds") de tamaño similar. Común: **`k = 5` o `k = 10`**.
2. **Entrenamiento y evaluación:** para cada fold, se entrena con `k−1` folds y el restante se usa como prueba. Se repite `k` veces rotando el fold de prueba.
3. **Promedio de resultados:** se promedian las métricas de las `k` iteraciones.

**Tipos comunes:**
- **k-Fold:** el dataset se divide en `k` partes.
- **Leave-One-Out (LOOCV):** caso extremo `k = n` (una observación por fold).
- **Stratified k-Fold:** mantiene la proporción de clases por fold (clasificación desbalanceada).

**Ventajas:** uso eficiente de los datos (cada dato sirve para train y test) y evaluación más **robusta** (se mide en múltiples particiones).

## 7. Hiperparámetros y su ajuste

- **Hiperparámetros:** características **externas** del modelo, **no se aprenden**; se definen antes de entrenar. Ej.: el valor `k` de K-means, la profundidad máxima de un árbol.
- **Parámetros:** características **internas**, estimadas durante el entrenamiento. Ej.: los coeficientes de una regresión lineal.

No se conocen a priori los mejores valores, y una configuración buena para un dataset puede no serlo para otro. Por eso el **hyperparameter tuning** (selección del conjunto óptimo) es esencial.

**Grid Search vs. Random Search** (mismo proceso: para cada combinación, aplicar en train → evaluar con CV → registrar puntaje → elegir el mejor):
- **Grid Search:** busca sobre **todas** las combinaciones de una grilla (búsqueda exhaustiva).
- **Random Search:** selecciona **al azar** un subset de combinaciones. Útil cuando evaluar todo es muy costoso.

**Cómo se implementa en sklearn** — cinco elementos:
1. Un **estimador** (el modelo).
2. Un **espacio de parámetros** (rango de valores).
3. Un **método de búsqueda** (GridSearch / RandomSearch).
4. Un **esquema de validación cruzada** (cantidad de folds).
5. La **métrica de evaluación** (criterio para elegir el mejor).
