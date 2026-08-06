# Apuntes Data Science — Digital House

> Documento maestro consolidado del programa. Combina **dos criterios en uno** por cada módulo:
> - **Apuntes** — la teoría explicada para entender y estudiar (como para un examen).
> - **Referencia técnica** — para cada técnica: cuándo aplicarla, para qué, hiperparámetros clave, cómo se evalúa y snippet de `scikit-learn`.
>
> Los títulos de módulo usan la **numeración del programa** (carpetas `NN-...`). Cada PDF del curso tiene además su conversión 1:1 en un `.md` al lado del archivo. Donde el material de origen tenía errores, se corrigen y se marcan con **Nota**.
>
> Estado de cobertura: Módulo 01 y 03 (apuntes); Módulo 04 (apuntes + referencia completa). Se completará la referencia del resto de módulos de forma incremental.

## Índice

- [Módulo 01 — Introducción a Machine Learning](#módulo-01--introducción-a-machine-learning)
  - [1. ¿Qué es Machine Learning?](#1-qué-es-machine-learning) · [2. Datasets, features y labels](#2-datasets-features-y-labels) · [3. Tipos de aprendizaje](#3-tipos-de-aprendizaje) · [4. Ciclo de vida de un proyecto de ML](#4-ciclo-de-vida-de-un-proyecto-de-ml) · [5. Preparación de datos y EDA](#5-preparación-de-datos-y-eda) · [6. Clasificadores lineales](#6-clasificadores-lineales) · [7. Árboles de decisión](#7-árboles-de-decisión) · [8. Métricas de clasificación](#8-métricas-de-clasificación) · [9. Aprendizaje no supervisado (panorama)](#9-aprendizaje-no-supervisado-panorama) · [10. Evaluación de modelos](#10-evaluación-de-modelos) · [11. XGBoost y ensambles](#11-xgboost-y-ensambles)
- [Módulo 02 — Desafío Profesional DS (Etapa 1)](#módulo-02--desafío-profesional-ds-etapa-1)
- [Módulo 03 — Modelado avanzado en Machine Learning](#módulo-03--modelado-avanzado-en-machine-learning)
  - [12. Regresión lineal](#12-regresión-lineal) · [13. Métricas de regresión](#13-métricas-de-regresión) · [14. Inferencia sobre los coeficientes](#14-inferencia-sobre-los-coeficientes)
- [Módulo 04 — Aprendizaje no supervisado](#módulo-04--aprendizaje-no-supervisado)
  - [15. Panorama del no supervisado](#15-panorama-del-no-supervisado) · [16. Clustering: K-means](#16-clustering-k-means) · [17. Clustering jerárquico](#17-clustering-jerárquico) · [18. DBSCAN](#18-dbscan-clustering-por-densidad) · [19. Evaluación de clusters](#19-evaluación-de-clusters) · [20. Selección de variables](#20-selección-de-variables) · [21. Regularización](#21-regularización-ridge-lasso-elastic-net) · [22. Criterios de selección de modelos (AIC/BIC)](#22-criterios-de-selección-de-modelos-aic--bic) · [23. Maldición de la dimensión](#23-la-maldición-de-la-dimensión) · [24. Reducción de dimensionalidad](#24-reducción-de-dimensionalidad-pca-lda-t-sne-ica)
- [Glosario rápido](#glosario-rápido)

---

## Módulo 01 — Introducción a Machine Learning

### 1. ¿Qué es Machine Learning?

**Machine Learning (ML)** es la rama de la Inteligencia Artificial que crea algoritmos capaces de **aprender una tarea a partir de datos**, sin programar reglas explícitas para cada caso.

| Concepto | Alcance |
|----------|---------|
| **Inteligencia Artificial (IA)** | Campo amplio: que una máquina imite comportamiento inteligente humano. |
| **Machine Learning (ML)** | Subconjunto de la IA: aprender de los datos / experiencia. |
| **Deep Learning** | Subconjunto del ML basado en redes neuronales profundas. |

**Aplicaciones típicas:** visión por computadora (reconocimiento facial, clasificación de imágenes), sistemas de recomendación, vehículos autónomos y robótica, salud (predicción y diagnóstico), NLP, detección de fraude.

### 2. Datasets, features y labels

- **Dataset:** colección estructurada de datos (filas = registros/observaciones, columnas = variables).
- **Features (características):** variables de entrada `X`.
- **Label / target (etiqueta):** variable a predecir `y`.

La **cantidad y calidad** de los datos determinan el techo de desempeño del modelo (*garbage in, garbage out*).

### 3. Tipos de aprendizaje

```
Aprendizaje Automático
├── Supervisado (datos etiquetados)
│   ├── Regresión      → target numérico continuo
│   └── Clasificación  → target categórico
└── No Supervisado (sin etiquetas)
    ├── Clustering              → agrupar por similitud
    └── Reducción de dimensión → comprimir/visualizar
```

Otras variantes: **semi-supervisado** (pocas etiquetas + muchos datos sin etiquetar) y **aprendizaje por refuerzo** (agente que aprende por recompensa).

| | Supervisado | No supervisado |
|---|---|---|
| Datos | Etiquetados (`X`, `y`) | Sin etiquetar (solo `X`) |
| Objetivo | Predecir `y` | Descubrir estructura oculta |
| Ejemplos | Regresión lineal, logística, árboles | K-Means, PCA |

### 4. Ciclo de vida de un proyecto de ML

1. **Definición del problema** — entender contexto y negocio, definir la pregunta, objetivos y métricas de éxito, variables de entrada y target.
2. **Preparación de datos** — recolección, EDA, limpieza, selección/ingeniería de features.
3. **Selección y entrenamiento** — elegir algoritmo según datos y requisitos; entrenar y ajustar hiperparámetros.
4. **Evaluación** — medir con métricas acordes al problema (clasificación vs regresión).
5. **Optimización y refinamiento** — tuning de hiperparámetros, feature engineering, regularización.
6. **Implementación en producción** — despliegue del modelo.
7. **Monitoreo y mantenimiento** — vigilar *drift* y reentrenar.

> **Nota:** las slides repetían el mismo texto en "Evaluación" y "Optimización". La optimización es iterativa: tras evaluar se ajustan hiperparámetros y features y se vuelve a evaluar.

### 5. Preparación de datos y EDA

- **EDA (Análisis Exploratorio de Datos):** estadística descriptiva + visualizaciones para entender distribución, relaciones, outliers y valores faltantes.
- **Selección de features:** quedarse con las variables más relevantes para el target (reduce ruido y overfitting).
- Tareas comunes: imputación de faltantes, encoding de categóricas, escalado/normalización, tratamiento de outliers, balanceo de clases.

### 6. Clasificadores lineales

Toman decisiones a partir de una **combinación lineal** de las features. Función de decisión:

```
f(x) = wᵀx + b
```

- `x`: vector de features · `w`: pesos · `b`: sesgo (bias).
- Clasificación binaria según el signo: `f(x) ≥ 0` → clase A; `f(x) < 0` → clase B.
- Un **hiperplano** es la frontera de decisión (recta en 2D, plano en 3D, etc.).

| Modelo | Idea clave |
|--------|-----------|
| **Perceptrón** | Clasificador lineal más simple; ajusta pesos según errores. |
| **Regresión logística** | Pese al nombre, **clasifica**: modela `P(y=1\|X)` con la función logística (sigmoide), acotando la salida a `[0,1]`. |
| **SVM** | Busca el hiperplano que **maximiza el margen** entre clases (optimización cuadrática). |

**¿Por qué no usar regresión lineal para clasificar?** Daría valores fuera de `[0,1]`, imposibles de interpretar como probabilidad. La logística resuelve esto con la sigmoide:

```
σ(z) = 1 / (1 + e^(−z))   con z = wᵀx + b
```

### 7. Árboles de decisión

Modelo predictivo con estructura jerárquica: **nodo raíz → nodos internos (preguntas) → ramas (respuestas) → hojas (predicción)**. Sirve para clasificación y regresión (**CART**).

**Cómo decide dónde ramificar:** prueba dividir por cada variable y elige la que produce subnodos más **homogéneos** (puros) respecto al target. Criterios de impureza para clasificación:

- **Índice de Gini** ∈ `[0, 1]`: `Gini(D) = 1 − Σ pᵢ²`.
- **Entropía** ∈ `[0, 1]` (binaria): `H(D) = −Σ pᵢ·log₂(pᵢ)`.

> **Corrección importante:** las slides decían *"a mayor índice de Gini, mayor homogeneidad"* — **es al revés**. Un nodo **puro/homogéneo tiene Gini = 0** (y entropía = 0); el valor crece cuanto **más mezcladas** están las clases. El árbol elige divisiones que **minimizan** la impureza (o maximizan la *ganancia de información*).

**Hiperparámetros** (controlan crecimiento y overfitting): profundidad máxima (`max_depth`), mínimo de observaciones para dividir (`min_samples_split`), mínimo en hoja (`min_samples_leaf`), criterio (gini/entropy).

| Ventajas | Desventajas |
|----------|-------------|
| Fáciles de interpretar | Tendencia al **sobreajuste** |
| Útiles en EDA (importancia de variables) | Inestables (pequeños cambios → árbol distinto) |
| Poca limpieza previa (toleran outliers/faltantes) | Menor precisión que ensambles/SVM |
| Soportan numéricas y categóricas | Pérdida de info al discretizar continuas |
| No paramétricos | |

El sobreajuste y la inestabilidad se mitigan con **ensambles** (Random Forest, Gradient Boosting / XGBoost).

### 8. Métricas de clasificación

**Matriz de confusión:**

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | TP (verdadero positivo) | FN (falso negativo) |
| **Real Negativo** | FP (falso positivo) | TN (verdadero negativo) |

| Métrica | Fórmula | Cuándo importa |
|---------|---------|----------------|
| **Accuracy** | `(TP+TN)/Total` | Exactitud global. Engañosa con clases desbalanceadas. |
| **Precisión** | `TP/(TP+FP)` | Cuando el costo de un FP es alto. |
| **Recall (sensibilidad)** | `TP/(TP+FN)` | Cuando el costo de un FN es alto (ej. diagnóstico). |
| **F1-score** | `2·(P·R)/(P+R)` | Media armónica P/R; equilibrio con datos desbalanceados. |

- **Curva ROC:** TPR (recall) vs FPR variando el umbral.
- **AUC:** área bajo la ROC. `1.0` = perfecto, `0.5` = azar.

> **Nota:** con clases desbalanceadas, priorizar **F1 / AUC / recall** según el caso, no accuracy.

### 9. Aprendizaje no supervisado (panorama)

Sin etiquetas: el objetivo es **descubrir patrones/estructura**. (Se desarrolla en profundidad en el [Módulo 04](#módulo-04--aprendizaje-no-supervisado).)

- **Clustering:** agrupar observaciones similares y separar las distintas. Usos: segmentación de mercado, detección de anomalías. Algoritmo típico: **K-Means**.
- **Reducción de dimensionalidad:** comprimir muchas features en pocas conservando la información relevante; útil para visualizar y reducir ruido. Algoritmo típico: **PCA**.

### 10. Evaluación de modelos

- **Overfitting (sobreajuste):** el modelo memoriza el train y generaliza mal (alta varianza).
- **Underfitting (subajuste):** el modelo es demasiado simple (alto sesgo).
- **Split train/test** y **validación cruzada (k-fold)** para estimar el desempeño real.
- **Trade-off sesgo-varianza:** buscar el punto de error de generalización mínimo.

### 11. XGBoost y ensambles

**Ensambles:** combinan varios modelos débiles para uno fuerte.

- **Bagging** (ej. Random Forest): entrena árboles en paralelo sobre muestras bootstrap y promedia → reduce varianza.
- **Boosting** (ej. XGBoost): entrena árboles **secuencialmente**, cada uno corrige los errores del anterior → reduce sesgo.

**XGBoost** (Extreme Gradient Boosting): implementación optimizada de gradient boosting; alto rendimiento, regularización incorporada y manejo eficiente de datos. Estándar de facto en problemas tabulares.

---

## Módulo 02 — Desafío Profesional DS (Etapa 1)

Proyecto integrador end-to-end. Etapas:

1. **Exploración visual de los datos** (EDA).
2. **Limpieza y transformación de datos.**

**Casos de negocio disponibles** (elegir uno y resolverlo aplicando todo el pipeline de ML):

| Caso | Dominio | Tipo de problema sugerido |
|------|---------|---------------------------|
| **Subtes** | Movilidad urbana (CABA) | Series temporales / regresión de demanda |
| **Airbnb** | Precios de alojamiento | Regresión de precio |
| **Cambio Climático** | Ambiental | Regresión / análisis de tendencias |
| **Diabetes** | Salud | Clasificación binaria |

> Los datasets de estos casos (ZIPs pesados) **no están versionados** en el repo por superar el límite de 100 MB de GitHub. Ver `README` para su origen.

---

## Módulo 03 — Modelado avanzado en Machine Learning

### 12. Regresión lineal

Predice una respuesta **cuantitativa** `Y` a partir de predictores `X`, asumiendo relación aproximadamente lineal.

**Simple:** `Y = β₀ + β₁·X + ε`
**Múltiple:** `Y = β₀ + β₁X₁ + … + βₚXₚ + ε`

- `β₀`: intercepto · `β₁…βₚ`: pendientes/coeficientes · `ε`: error.
- **Entrenar = estimar los coeficientes** (β̂) que mejor ajustan los datos.
- **Residuo:** `eᵢ = yᵢ − ŷᵢ`.
- Método de **mínimos cuadrados (OLS):** minimiza la suma de cuadrados de los residuos
  `RSS = Σ (yᵢ − ŷᵢ)²`.

### 13. Métricas de regresión

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **MAE** | `(1/n)·Σ\|yᵢ − ŷᵢ\|` | Error medio absoluto; misma unidad que `y`; robusto a outliers. |
| **MSE** | `(1/n)·Σ(yᵢ − ŷᵢ)²` | Penaliza más los errores grandes. |
| **RMSE** | `√MSE` | Como MSE pero en la unidad de `y`; más interpretable. |
| **R²** | `1 − RSS/TSS` | Proporción de varianza explicada por el modelo. |

**Descomposición de la varianza:** `TSS = ESS + RSS`
(Total = Explicada + Residual).

**Interpretación de R²:**
- `R² = 1` → ajuste perfecto, sin error.
- `R² = 0` → no mejor que predecir la media.
- `R² < 0` → peor que predecir la media.

> **Nota:** R² es invariante a la escala de `y`, pero el intercepto, MAE, MSE y RMSE **no** lo son (dependen de las unidades). RMSE = √MSE es la más interpretable por estar en la unidad de `y`.

### 14. Inferencia sobre los coeficientes

**Test de significación individual** para `βⱼ`:

- **H₀:** `βⱼ = 0` (no hay relación entre `Xⱼ` e `Y`).
- **H₁:** `βⱼ ≠ 0` (hay relación).

Si `β₁ = 0`, el modelo se reduce a `Y = β₀ + ε` y `X` no aporta. Se evalúa con el **p-value**:

- **p-value chico** (< 0,05 típicamente) → se rechaza H₀: hay evidencia de relación.
- También se usan **intervalos de confianza** para los coeficientes.

> Este módulo cubre además validación cruzada, sesgo-varianza, regularización, ensambles, SVM/kernels y tuning. Ver la [guía de estudio del módulo 03](03-modelado-avanzado-en-machine-learning/GUIA-ESTUDIO.md) para el detalle con diagramas y tablas de hiperparámetros.

---

## Módulo 04 — Aprendizaje no supervisado

> Curso "ML3". Combina **apuntes** (teoría) + **referencia técnica** (cuándo/hiperparámetros/snippets). Detalle por clase en el [README del módulo](04-aprendizaje-no-supervisado/README.md).

### 15. Panorama del no supervisado

Sin variable objetivo: el objetivo es **descubrir estructura** en los datos. Dos familias principales + una transversal:

| Familia | Objetivo | Técnicas |
|---------|----------|----------|
| **Clustering** | Agrupar observaciones similares | K-means, jerárquico, DBSCAN |
| **Reducción de dimensionalidad** | Bajar nº de variables preservando información | PCA, LDA, t-SNE, ICA |
| **Selección de variables** (transversal) | Elegir un subconjunto de features | Filtros, RFE, embedded, regularización |

**Extracción vs selección de características:**
- **Extracción** (PCA, LDA, t-SNE, ICA): **crea nuevas** variables combinando las originales.
- **Selección** (filtros, RFE, embedded): **elige un subconjunto** de las originales sin transformarlas.

> **Regla de oro del módulo:** estandarizar (`StandardScaler`) antes de todo método basado en **distancias** (K-means, DBSCAN, K-NN) o en **magnitud de coeficientes/varianza** (PCA, Ridge, Lasso).

### 16. Clustering: K-means

**Apuntes.** Particiona los datos en **k** clusters minimizando la **inercia** (suma de distancias² de cada punto a su centroide). Iterativo: inicializa k centroides → asigna cada punto al más cercano → recalcula centroides como la media → repite hasta converger.

**Referencia técnica.**

| | |
|--|--|
| **Cuándo usar** | Sabés (o estimás) k; grupos aproximadamente **esféricos** y de tamaño similar; dataset grande (escala bien). |
| **Cuándo NO** | Formas arbitrarias, densidades/tamaños muy distintos, outliers (los arrastra). |
| **Hiperparámetros** | `n_clusters` (**el** clave), `init` (`k-means++`), `n_init`, `max_iter`. |
| **Escala** | Sensible → estandarizar antes. |
| **Evaluación** | Codo (inercia), silueta, Davies-Bouldin (§19). |

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
km.inertia_   # WCSS para el método del codo
```

### 17. Clustering jerárquico

**Apuntes.** Construye una jerarquía fusionando iterativamente los clusters más cercanos; se visualiza con un **dendrograma** (la altura de cada fusión = distancia entre grupos). Se corta a una altura para obtener k clusters.

**Referencia técnica.**

| | |
|--|--|
| **Cuándo usar** | No sabés cuántos clusters hay (cortás a posteriori); querés ver la estructura jerárquica; dataset chico/mediano. |
| **Cuándo NO** | Datasets grandes (costo O(n²) o peor). |
| **Hiperparámetros** | `n_clusters` o `distance_threshold`, `linkage`, `metric`. |
| **Enlaces (`linkage`)** | simple (min), completo (max), promedio (average), centroide, **Ward** (minimiza varianza intra; el más usado). |

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
Z = linkage(X_scaled, method="ward", metric="euclidean")
labels = fcluster(Z, t=3, criterion="maxclust")
```

### 18. DBSCAN (clustering por densidad)

**Apuntes.** Define clusters como **regiones densas** separadas por regiones vacías. No requiere fijar k, detecta clusters de forma arbitraria y marca **outliers** (etiqueta `-1`).

- **Core:** ≥ `min_samples` puntos dentro del radio `eps`.
- **Border:** dentro de `eps` de un core, pero no es core.
- **Noise:** ni core ni border → outlier.

**Referencia técnica.**

| | |
|--|--|
| **Cuándo usar** | Formas arbitrarias; hay ruido/outliers a detectar; no querés fijar k. |
| **Cuándo NO** | Densidad muy variable entre clusters; alta dimensión. |
| **Hiperparámetros** | `eps` (ε, radio), `min_samples` (MinPts). |
| **Elegir `eps`** | Gráfico k-distancia (k = min_samples): buscar el codo. |

```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)  # -1 = ruido
```

**Comparativa de clustering:**

| Método | Nº clusters | Formas | Ruido | Escala |
|--------|:-----------:|--------|:-----:|--------|
| **K-means** | Hay que fijarlo | Esféricas | Sensible | Rápido |
| **Jerárquico** | Se decide al cortar | Según enlace | Según método | Pesado en grandes |
| **DBSCAN** | Automático | **Arbitrarias** | **Detecta** | Sensible a `eps` |

### 19. Evaluación de clusters

Sin etiquetas → métricas **internas** (geometría de los clusters).

| Método | Qué mide | Cómo se lee | Rango |
|--------|----------|-------------|-------|
| **Método del codo** | Inercia / WCSS vs k | Elegir el k del "codo" (mejora marginal) | Inercia ↓ con k |
| **Silueta** | `s = (b − a)/max(a,b)`; cohesión vs separación | +1 bien agrupado; ~0 frontera; negativo mal asignado | [−1, +1] |
| **Davies-Bouldin** | Dispersión intra / separación inter | **Menor = mejor** | ≥ 0 |

> **Buena práctica:** validar con **codo Y silueta**; suelen sugerir distinto k (ej. Iris: codo→3, silueta→2) y se decide con criterio del dominio.

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
silhouette_score(X_scaled, labels)       # más alto mejor
davies_bouldin_score(X_scaled, labels)   # más bajo mejor
```

### 20. Selección de variables

Elegir un **subconjunto** de las features originales. Mejora rendimiento, interpretabilidad, reduce overfitting y acelera el entrenamiento.

**Filtros (filter)** — medida estadística, sin entrenar modelo (rápidos, antes de modelar):

| Método | Problema | Idea |
|--------|----------|------|
| **Correlación (Pearson)** | Regresión | features con coef. cercano a ±1 con el target |
| **Chi-cuadrado (χ²)** | Clasificación, categóricas | mayor dependencia feature-clase = más relevante |
| **ANOVA (F-test)** | Clasificación, numéricas | mayor F = mejor separa clases |
| **Coeficientes lineales** | Regresión/logística | mayor \|coef\| = más influyente |

**Wrapper — RFE (Eliminación Recursiva):** entrena un modelo, elimina la feature menos importante, reentrena y repite. Preciso pero **caro** (entrena en cada iteración).

**Embedded:** la selección ocurre dentro del entrenamiento (árboles/Random Forest vía `feature_importances_`). Mantiene las features originales.

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)   # filtro ANOVA
```

| Criterio | Filtros | RFE |
|----------|---------|-----|
| Depende de un modelo | No | **Sí** |
| Costo | Bajo | Alto (iterativo) |
| Cuándo | Screening rápido | Optimizar modelo final |

### 21. Regularización (Ridge, Lasso, Elastic Net)

Añade una **penalización** al costo: `Costo = RSS + α·penalización`. El hiperparámetro **α (o λ)** regula la fuerza y se elige por **cross-validation**. Estandarizar antes (la penalización depende de la escala).

| Método | Norma | ¿Anula coeficientes? | Cuándo usar |
|--------|:-----:|:--------------------:|-------------|
| **Ridge** | L2 (`Σβ²`) | **No** (los achica) | Colinealidad; conservar todas las variables |
| **Lasso** | L1 (`Σ\|β\|`) | **Sí** (a cero exacto) | **Selección automática**; modelos dispersos |
| **Elastic Net** | L1 + L2 | Sí (parcial) | Combina ambas; **2 hiperparámetros** (λ, α) |

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_scaled, y)   # elige alpha por CV
lasso.coef_    # los que quedan en 0 fueron descartados
```

### 22. Criterios de selección de modelos (AIC / BIC)

Comparan modelos equilibrando **ajuste vs complejidad** (penalizan el nº de parámetros). **Menor = mejor.**

| | AIC | BIC |
|--|-----|-----|
| **Fórmula** | `2k − 2·ln(L)` | `ln(n)·k − 2·ln(L)` |
| **Penalización** | `2` por parámetro (fija) | `ln(n)` (crece con la muestra) |
| **Tendencia** | Admite más complejidad | Más conservador (modelos simples) |

`k` = nº parámetros, `n` = nº observaciones, `L` = máxima verosimilitud. Usos: comparar modelos de regresión, ARIMA, o el nº de componentes en un **GMM**.

### 23. La maldición de la dimensión

Problemas en **alta dimensión**: los datos se **dispersan**, las distancias se vuelven **uniformes** y pierden significado (degradan K-means/K-NN); el costo crece exponencialmente; sube el riesgo de **overfitting**; la visualización se vuelve imposible.

**Mitigaciones:** reducción de dimensionalidad (§24), selección de variables (§20), regularización (§21). Es el puente entre selección y reducción.

### 24. Reducción de dimensionalidad (PCA, LDA, t-SNE, ICA)

**Crean** nuevas variables que preservan información (distinto de seleccionar).

**PCA (Análisis de Componentes Principales)** — lineal, **no supervisado**. Encuentra las direcciones de **máxima varianza** (componentes principales), ortogonales entre sí.
- **Pasos:** centrar → matriz de covarianza → valores/vectores propios → ordenar por valor propio → proyectar sobre los top componentes.
- **Nº de componentes:** scree plot (codo en los valores propios) o varianza explicada acumulada (ej. retener 95%).
- **Cuándo:** conservar varianza, descorrelacionar features, visualizar 2-3D, preprocesar. Sensible a escala → estandarizar; `fit` solo con train.

**LDA (Análisis Discriminante Lineal)** — lineal, **supervisado**. Busca la proyección que **maximiza la separación entre clases**. Genera hasta **(nº clases − 1)** componentes.

**t-SNE** — no lineal; preserva la **estructura de vecinos locales** minimizando la divergencia **KL**. Ideal para **visualización** en 2-3D (no para preprocesar). Costoso en datasets grandes; la estructura **global** (distancias entre clusters) puede no ser fiable.

**UMAP** — no lineal (geometría/topología); construye un **grafo de vecindad** y preserva estructura **local y global**. **Más rápido y escalable que t-SNE**; ajustable con `n_neighbors` / `min_dist`. No viene en sklearn: `pip install umap-learn`.

**ICA** — separa componentes **estadísticamente independientes** (ej. separación de fuentes de audio).

| Técnica | Lineal | Supervisada | Maximiza / preserva | Uso principal |
|---------|:------:|:-----------:|----------|---------------|
| **PCA** | Sí | No | Varianza | Compresión / preprocesamiento |
| **LDA** | Sí | **Sí** | Separación entre clases | Preproc. para clasificación |
| **t-SNE** | No | No | Estructura **local** | **Visualización** (datasets chicos/medianos) |
| **UMAP** | No | No | Estructura **local y global** | **Visualización** (datasets grandes) |
| **ICA** | Sí | No | Independencia estadística | Separación de fuentes |

**Elección rápida:** preprocesar/comprimir → **PCA**; separar clases con etiquetas → **LDA**; visualizar estructura local (dataset chico) → **t-SNE**; visualizar dataset grande preservando local + global → **UMAP**.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)                 # retener 95% de la varianza
X_red = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
X_lda = LDA(n_components=2).fit_transform(X, y)   # supervisado (usa y)

from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, perplexity=30).fit_transform(X_scaled)  # visualización

import umap                                   # pip install umap-learn
X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X_scaled)
```

**Tabla maestra — ¿qué técnica uso?**

| Necesito… | Técnica | § |
|-----------|---------|:-:|
| Agrupar, sé k, grupos esféricos | K-means | 16 |
| Agrupar y ver estructura jerárquica | Jerárquico | 17 |
| Agrupar formas raras + detectar outliers | DBSCAN | 18 |
| Elegir nº óptimo de clusters | Codo + Silueta | 19 |
| Descartar features rápido, sin modelo | Filtros (χ²/ANOVA) | 20 |
| Mejor subconjunto para un modelo | RFE | 20 |
| Selección automática en un lineal | Lasso / Elastic Net | 21 |
| Reducir dimensión conservando varianza | PCA | 24 |
| Reducir dimensión separando clases | LDA | 24 |
| Visualizar alta dimensión en 2D (dataset chico) | t-SNE | 24 |
| Visualizar alta dimensión en 2D (dataset grande) | UMAP | 24 |
| Comparar modelos ajuste/complejidad | AIC / BIC | 22 |

---

## Glosario rápido

| Término | Definición |
|---------|-----------|
| **Feature** | Variable de entrada (`X`). |
| **Label / Target** | Variable a predecir (`y`). |
| **Hiperparámetro** | Configuración fijada antes de entrenar (ej. `max_depth`, `n_clusters`). |
| **Overfitting** | Memoriza el train, generaliza mal (alta varianza). |
| **Underfitting** | Demasiado simple, no captura el patrón (alto sesgo). |
| **Hiperplano** | Frontera de decisión lineal. |
| **Impureza (Gini/Entropía)** | Mezcla de clases en un nodo; 0 = puro. |
| **Ensamble** | Combinación de varios modelos (bagging/boosting). |
| **OLS** | Mínimos cuadrados ordinarios. |
| **RSS / TSS / ESS** | Suma de cuadrados Residual / Total / Explicada. |
| **Inercia / WCSS** | Suma de distancias² de cada punto a su centroide; la minimiza K-means. |
| **Silueta** | `(b−a)/max(a,b)`; cohesión vs separación de cada punto. |
| **Core / border / noise** | En DBSCAN: punto denso / de borde / outlier. |
| **Componente principal** | Dirección de máxima varianza (PCA); combinación lineal ortogonal de las variables. |
| **Valor / vector propio** | Dirección principal (vector) y cuánta varianza explica (valor). |
| **Scree plot** | Gráfico de valores propios ordenados; su codo indica cuántos componentes retener. |
| **Varianza explicada** | Proporción de la varianza total capturada por un componente. |
| **Norma L1 / L2** | `Σ\|β\|` / `√Σβ²`; base de Lasso / Ridge. |
| **AIC / BIC** | Criterios de selección de modelo (ajuste vs complejidad); menor = mejor. |
| **Maldición de la dimensión** | Degradación de distancias y algoritmos en alta dimensión. |
