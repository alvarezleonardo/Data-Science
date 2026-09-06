# Apuntes Data Science — Digital House

> Documento maestro consolidado del programa. Combina **dos criterios en uno** por cada módulo:
> - **Apuntes** — la teoría explicada para entender y estudiar (como para un examen).
> - **Referencia técnica** — para cada técnica: cuándo aplicarla, para qué, hiperparámetros clave, cómo se evalúa y snippet de `scikit-learn`.
>
> Los títulos de módulo usan la **numeración del programa** (carpetas `NN-...`). Cada PDF del curso tiene además su conversión 1:1 en un `.md` al lado del archivo. Donde el material de origen tenía errores, se corrigen y se marcan con **Nota**.
>
> Estado de cobertura: Módulos 01, 03 y 04 con **ambos criterios** (apuntes + referencia técnica). Módulo 06 con las primeras clases documentadas (fundamentos biológicos, historia, perceptrón y su entrenamiento, limitaciones, implementación con scikit-learn y perceptrón multicapa); el módulo sigue en curso y se irá completando con funciones de activación, funciones de pérdida, regularización, optimización y el detalle de backpropagation. El resto se completará a medida que avance el programa.

## Índice

- [Módulo 01 — Introducción a Machine Learning](#módulo-01--introducción-a-machine-learning)
  - [1. ¿Qué es Machine Learning?](#1-qué-es-machine-learning) · [2. Datasets, features y labels](#2-datasets-features-y-labels) · [3. Tipos de aprendizaje](#3-tipos-de-aprendizaje) · [4. Ciclo de vida de un proyecto de ML](#4-ciclo-de-vida-de-un-proyecto-de-ml) · [5. Preparación de datos y EDA](#5-preparación-de-datos-y-eda) · [6. Clasificadores lineales](#6-clasificadores-lineales) · [7. Árboles de decisión](#7-árboles-de-decisión) · [8. Métricas de clasificación](#8-métricas-de-clasificación) · [9. Aprendizaje no supervisado (panorama)](#9-aprendizaje-no-supervisado-panorama) · [10. Evaluación de modelos](#10-evaluación-de-modelos) · [11. XGBoost y ensambles](#11-xgboost-y-ensambles) · [Referencia técnica](#referencia-técnica--módulo-01)
- [Módulo 02 — Desafío Profesional DS (Etapa 1)](#módulo-02--desafío-profesional-ds-etapa-1)
- [Módulo 03 — Modelado avanzado en Machine Learning](#módulo-03--modelado-avanzado-en-machine-learning)
  - [12. Regresión lineal](#12-regresión-lineal) · [13. Métricas de regresión](#13-métricas-de-regresión) · [14. Inferencia sobre los coeficientes](#14-inferencia-sobre-los-coeficientes) · [Referencia técnica](#referencia-técnica--módulo-03)
- [Módulo 04 — Aprendizaje no supervisado](#módulo-04--aprendizaje-no-supervisado)
  - [15. Panorama del no supervisado](#15-panorama-del-no-supervisado) · [16. Clustering: K-means](#16-clustering-k-means) · [17. Clustering jerárquico](#17-clustering-jerárquico) · [18. DBSCAN](#18-dbscan-clustering-por-densidad) · [19. Evaluación de clusters](#19-evaluación-de-clusters) · [20. Selección de variables](#20-selección-de-variables) · [21. Regularización](#21-regularización-ridge-lasso-elastic-net) · [22. Criterios de selección de modelos (AIC/BIC)](#22-criterios-de-selección-de-modelos-aic--bic) · [23. Maldición de la dimensión](#23-la-maldición-de-la-dimensión) · [24. Reducción de dimensionalidad](#24-reducción-de-dimensionalidad-pca-lda-t-sne-ica)
- [Módulo 06 — Fundamentos de redes neuronales](#módulo-06--fundamentos-de-redes-neuronales)
  - [25. Fundamentos biológicos](#25-fundamentos-biológicos) · [26. Historia de las redes neuronales](#26-historia-de-las-redes-neuronales) · [27. El perceptrón: estructura y fórmulas](#27-el-perceptrón-estructura-y-fórmulas) · [28. Entrenamiento del perceptrón: ejemplo compuerta AND](#28-entrenamiento-del-perceptrón-ejemplo-compuerta-and) · [29. Limitaciones del perceptrón](#29-limitaciones-del-perceptrón) · [30. Implementación con scikit-learn](#30-implementación-con-scikit-learn) · [31. Perceptrón multicapa (MLP)](#31-perceptrón-multicapa-mlp) · [Referencia técnica](#referencia-técnica--módulo-06)
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

### Referencia técnica — Módulo 01

**Modelos supervisados: cuándo usar cada uno + hiperparámetros clave**

| Modelo | Cuándo usar | Hiperparámetros clave |
|--------|-------------|-----------------------|
| **Regresión logística** | Baseline de clasificación; interpretable; querés probabilidades | `C` (inverso de la regularización), `penalty` (l1/l2), `class_weight` |
| **SVM** (`SVC`) | Márgenes claros; fronteras no lineales (kernels); datasets chicos/medianos | `C`, `kernel` (linear/rbf), `gamma` |
| **Árbol de decisión** | Interpretabilidad, no linealidad, EDA de importancia | `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion` |
| **Random Forest** | Robusto, poca config, da importancia de features | `n_estimators`, `max_depth`, `max_features` |
| **XGBoost** | Máxima performance en datos tabulares | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

**Cómo se evalúa** (ver §8): matriz de confusión, `accuracy` / `precision` / `recall` / `F1`, `ROC-AUC`. Con clases desbalanceadas priorizar F1/AUC/recall. Estimar el desempeño real con **split train/test** + **validación cruzada** (§10).

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42).fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te)))
cross_val_score(clf, X, y, cv=5, scoring="f1_macro")     # desempeño robusto
clf.feature_importances_                                  # importancia de variables
```

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

### Referencia técnica — Módulo 03

**Flujo de trabajo típico + herramientas**

| Necesito… | Herramienta | Notas |
|-----------|-------------|-------|
| Ajustar una regresión | `LinearRegression` (OLS) | Sin hiperparámetros; base de comparación |
| Evaluar regresión | `mean_absolute_error`, `mean_squared_error`, `r2_score` | RMSE = `√MSE`; R² invariante a escala de `y` (§13) |
| Estimar desempeño real | `cross_val_score`, `KFold` | Promediar métricas sobre los folds; evita sobreestimar |
| Buscar hiperparámetros | `GridSearchCV` (exhaustivo) / `RandomizedSearchCV` (muestreo) | Combinar con CV; `scoring` acorde al problema |
| Encadenar preproceso + modelo | `Pipeline` | Evita fuga de datos (el `fit` del scaler queda dentro de cada fold) |
| Regularizar / seleccionar variables | `RidgeCV`, `LassoCV`, `ElasticNetCV` | Ver §21 (Módulo 04) para el detalle |

**Cómo se evalúa:** para regresión, **RMSE** (interpretable, en la unidad de `y`) y **R²** (varianza explicada). Vigilar el **trade-off sesgo-varianza** (§10): comparar error de train vs test para detectar over/underfitting.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
grid = GridSearchCV(pipe, {"model__alpha": [0.1, 1, 10]},
                    cv=KFold(5, shuffle=True, random_state=42),
                    scoring="neg_root_mean_squared_error")
grid.fit(X_tr, y_tr)
pred = grid.predict(X_te)
print("RMSE:", np.sqrt(mean_squared_error(y_te, pred)), "R2:", r2_score(y_te, pred))
```

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

## Módulo 06 — Fundamentos de redes neuronales

> Curso de redes neuronales. Documentado hasta la Clase 11 del programa: perceptrón, su entrenamiento manual con la compuerta AND, limitaciones, implementación con scikit-learn (`Perceptron`) y perceptrón multicapa (`MLPClassifier`, `MLPRegressor`). El módulo sigue en curso: faltan funciones de activación, grafos/capa densa, funciones de pérdida, regularización, optimización/descenso por gradiente y el detalle interno de backpropagation. Ver el [programa completo del módulo](06-fundamentos-de-redes-neuronales/teoria/0%20-%20Programa%20del%20módulo.md).

### 25. Fundamentos biológicos

**Apuntes.** Las **redes neuronales artificiales (RNA)** están inspiradas en el cerebro humano (~86 mil millones de neuronas conectadas por **sinapsis**). Cada neurona biológica **recibe** señales por las **dendritas**, las **procesa** en el **cuerpo celular (soma)** y las **transmite** por el **axón**.

Paralelismo biológico → artificial:

| Elemento biológico | Elemento artificial |
|---|---|
| Dendritas | Entradas (x₁, x₂, …, xₙ) |
| Fuerza de la sinapsis | Pesos (w₁, w₂, …, wₙ) |
| Cuerpo celular (integración) | Sumatoria ponderada (Σ) |
| Umbral de disparo | Función de activación (Φ) |
| Axón (señal transmitida) | Salida (y) |

Las neuronas artificiales se organizan en **capas**: capa de entrada, una o más **capas ocultas**, y capa de salida. El aprendizaje biológico fortalece/debilita sinapsis; el de una RNA **ajusta pesos** — analogía conceptual, no una réplica: el mecanismo real (backpropagation + optimización numérica) es matemáticamente distinto, y una RNA profunda consume muchísima más energía que el cerebro para tareas equivalentes.

### 26. Historia de las redes neuronales

**Apuntes.** Recorrido por décadas, con avances y un freno prolongado:

| Período | Hito | Impacto |
|---|---|---|
| 1940 (1943) | Modelo matemático de neurona (**McCulloch-Pitts**) | Sienta las bases teóricas: neuronas que hacen cálculos lógicos con entradas binarias |
| 1950 (1958) | **Perceptrón** (Rosenblatt) | Primer modelo capaz de aprender a clasificar datos linealmente separables |
| 1960 (1969) | Libro *"Perceptrons"* (**Minsky y Papert**) | Expone las limitaciones del perceptrón simple ante problemas no lineales → **invierno de la IA** (cae el financiamiento e interés) |
| 1980 | **Backpropagation** | Reactiva el campo: permite entrenar redes multicapa ajustando pesos para minimizar el error |
| 1990-2000 | **CNN** (datos con estructura espacial, imágenes) y **RNN** (datos secuenciales) | Especialización de arquitecturas según el tipo de dato |
| Era moderna | **Deep learning** | Datos masivos + poder de cómputo + mejores algoritmos → redes de decenas/cientos de capas |

> **Nota:** las etiquetas de década (1940, 1950, 1960) son aproximaciones del material original; los años puntuales son 1943 (McCulloch-Pitts), 1958 (Rosenblatt) y 1969 (Minsky y Papert).

### 27. El perceptrón: estructura y fórmulas

**Apuntes.** El **perceptrón** (Rosenblatt, 1957/58) es un algoritmo de **aprendizaje supervisado** para **clasificación binaria**, el modelo más simple de red neuronal. Es un **clasificador lineal**: solo puede aprender correctamente cuando las clases son **linealmente separables** (se pueden dividir con una única recta/hiperplano).

**Estructura:**

- **Entradas** `x1, x2, …, xn`.
- **Pesos** `w1, w2, …, wn`: importancia relativa de cada entrada.
- **Umbral / bias** `θ`: término independiente.
- **Suma ponderada** `Σ`, **función de activación** `Φ` y **salida** `y`.

Flujo: `x1…xn` (con pesos `w1…wn`) y `θ` → suma ponderada `z` → activación `Φ` → salida `y`.

**Fórmulas:**

```
Suma ponderada:         z = w1*x1 + w2*x2 + ... + wn*xn + θ   =   Σ (i=1 a n) wi*xi + θ

Función de activación (escalón):
Φ(z) = 1  si z > 0
Φ(z) = 0  en caso contrario

Salida:                 y = Φ(z)

Actualización de pesos:  Δwi = n * (y_real - y) * xi     →  wi_nuevo = wi_anterior + Δwi
Actualización del umbral: Δθ = n * (y_real - y)          →  θ_nuevo  = θ_anterior + Δθ
```

`n` (eta) es la **tasa de aprendizaje**: regula cuánto se ajusta cada peso en cada actualización. Ambas actualizaciones (pesos y umbral) se aplican en cada iteración en la que la predicción `y` difiere del valor real `y_real`; si coinciden, el ajuste es nulo (`Δw = Δθ = 0`).

**Ejemplo canónico — compuerta lógica AND** (`x1 AND x2`, tabla de verdad con un único caso positivo en `(1,1)`): al graficar los 4 puntos, `(1,1)` queda separado del resto por una recta → es **linealmente separable**, condición necesaria para que el perceptrón la aprenda.

### 28. Entrenamiento del perceptrón: ejemplo compuerta AND

**Apuntes.** Entrenamiento manual, paso a paso, de un perceptrón para aprender `x1 AND x2`.

**Parámetros iniciales:** `w1 = 0.381`, `w2 = 0.245`, `θ = 0.196`, tasa de aprendizaje `n = 0.045`.

El procedimiento se repite en cada iteración recorriendo las 4 filas de la tabla de verdad `(0,0)→0`, `(0,1)→0`, `(1,0)→0`, `(1,1)→1`, calculando `z`, aplicando la función escalón para obtener `y`, y actualizando `wi` y `θ` cuando `y ≠ y_real`.

**Resumen de las 6 iteraciones hasta la convergencia:**

| Iteración | Parámetros al cierre | ¿Hubo error en alguna fila? |
|:--:|---|---|
| Inicio | `w1=0.381`, `w2=0.245`, `θ=0.196` | — |
| 1ª | `w1=0.336`, `w2=0.2`, `θ=0.061` | Sí (filas 1, 2 y 3) |
| 2ª | `w1=0.291`, `w2=0.155`, `θ=-0.074` | Sí (filas 1, 2 y 3) |
| 3ª | `w1=0.246`, `w2=0.11`, `θ=-0.164` | Sí (filas 2 y 3) |
| 4ª | `w1=0.201`, `w2=0.11`, `θ=-0.209` | Sí (solo fila 3) |
| 5ª | `w1=0.201`, `w2=0.11`, `θ=-0.209` (sin cambios) | No — el perceptrón ya clasifica bien las 4 filas |
| 6ª (verificación) | `w1=0.201`, `w2=0.11`, `θ=-0.209` | No — error `0` confirmado en las 4 filas; converge |

**Pesos finales:** `w1 = 0.201`, `w2 = 0.11`, `θ (bias) = -0.209`. Con estos valores, `y = Φ(w1*x1 + w2*x2 + θ)` da `0` para `(0,0)`, `(0,1)` y `(1,0)`, y `1` para `(1,1)` — reproduce exactamente la compuerta AND.

> **Nota:** en la 5ª iteración ya no hubo ningún error, pero recién se confirma la convergencia con una 6ª pasada de verificación (error `e = y_real - y = 0` en las 4 filas) antes de dar por finalizado el entrenamiento.

### 29. Limitaciones del perceptrón

**Apuntes.**

- **Separabilidad lineal:** el perceptrón simple **solo** resuelve problemas **linealmente separables**. El caso clásico que no puede resolver es el **XOR**, cuyas clases no se pueden separar con una única recta/hiperplano.
- **Convergencia no garantizada:** si los datos no son linealmente separables, el algoritmo puede no converger nunca en un número finito de pasos.
- **Capacidad limitada:** poca capacidad para capturar relaciones y patrones complejos (al ser un modelo lineal de una sola capa).
- **Ajuste de hiperparámetros:** aunque tiene menos hiperparámetros que modelos más complejos, calibrar la tasa de aprendizaje sigue siendo un desafío (muy alta → inestabilidad/oscilación; muy baja → convergencia lenta).

Esta limitación (XOR) es históricamente la que originó el "invierno de la IA" (§26): se resuelve con perceptrones **multicapa** (MLP) y **backpropagation**, contenido posterior del módulo.

### 30. Implementación con scikit-learn

**Apuntes.**

- **Scikit-Learn** es una de las librerías más usadas del ecosistema Python para **aprendizaje automático**.
- Ofrece una **amplia gama de algoritmos y utilidades** que cubren **todo el flujo de trabajo**: desde la preparación de los datos hasta la evaluación del modelo.
- **Se integra** con el resto del ecosistema Python, incluidas las librerías de aprendizaje profundo como **TensorFlow** y **PyTorch**.

Aplicado a este módulo: el entrenamiento manual de la compuerta AND (§28) queda encapsulado en `sklearn.linear_model.Perceptron`, que implementa la misma regla delta detrás de la API uniforme `fit` / `predict` (ver ejemplo en la referencia técnica de más abajo). La implementación manual sirve para entender la mecánica; scikit-learn, para trabajar en la práctica con validación, métricas y preprocesamiento integrados. Para redes profundas (varias capas, backpropagation) se pasa a TensorFlow o PyTorch.

### 31. Perceptrón multicapa (MLP)

**Apuntes.**

Un **perceptrón multicapa (MLP)** es una red neuronal artificial que aborda problemas complejos de **clasificación** y **regresión**. A diferencia del perceptrón simple, que solo resuelve problemas **linealmente separables** (§29), el MLP maneja **relaciones no lineales** entre las características de entrada y la salida.

Los cuatro puntos de la slide:

- El entrenamiento se realiza mediante el algoritmo de **retropropagación** (backpropagation).
- Su capacidad para aprender relaciones complejas viene de la **estructura de capas** y de las **funciones de activación no lineales**.
- Las redes multicapa son **sensibles a la calidad y la naturaleza de los datos** de entrada.
- Necesitan **muchos datos** para entrenar eficazmente y tienen **riesgo de sobreajuste**.

**Arquitectura.**

| Capa | Rol |
|---|---|
| **Entrada** | una neurona por característica; no calcula, solo recibe |
| **Ocultas** (1 o más) | dan la capacidad de representar relaciones no lineales; cada neurona hace `z = Σ wᵢ·xᵢ + b` y le aplica una activación no lineal |
| **Salida** | una neurona en regresión; una por clase en clasificación |

Sobre esa arquitectura corren dos procesos:

- **Forward propagation:** los datos de entrada se transforman a través de las capas hasta producir la **salida final**.
- **Backpropagation:** ajuste de **pesos y sesgos** para **minimizar la función de pérdida**. Es la generalización a varias capas de la regla delta del entrenamiento manual (§28).

**Por qué la activación oculta debe ser no lineal.** Es el punto que justifica toda la arquitectura: si las capas ocultas solo hicieran la suma ponderada, la composición de varias capas lineales **seguiría siendo lineal** y la red colapsaría al equivalente de un perceptrón simple, con la misma limitación. La no linealidad (ReLU, tanh, logística) es lo que hace que apilar capas agregue capacidad real, y con eso resolver el **XOR**.

**Qué se paga respecto del perceptrón simple.**

| | Perceptrón simple | MLP |
|---|---|---|
| Problemas | solo linealmente separables | también no lineales |
| Interpretabilidad | alta: 2 pesos y un sesgo legibles | baja: miles de pesos sin lectura directa |
| Datos necesarios | pocos | muchos |
| Sobreajuste | bajo (modelo rígido) | alto: requiere regularización (`alpha`) y validación |
| Escalado de entradas | tolerable | **necesario** |
| Costo de entrenamiento | trivial | significativo |

> **Nota (verificado en notebook):** sobre Iris con 2 atributos y las 3 clases, el perceptrón simple llega a **76,7%** de exactitud y el MLP a **93,3%** — la diferencia es exactamente el solapamiento entre *versicolor* y *virginica*, que ninguna recta separa. En el mismo experimento, una red de **20** neuronas iguala a una de **250** usando dos órdenes de magnitud menos parámetros, mientras que una de **5** se queda corta (80%): el tamaño de la red es un hiperparámetro **a buscar**, no a maximizar.

### 32. Funciones de activación

**Apuntes.**

La **función de activación** determina la salida de una neurona. Sin ella, la red es una **combinación lineal de las entradas** y apilar capas no agrega nada (§31). Sus cuatro roles según la slide: introducen **no linealidad**, deciden si la neurona **se activa** (pasa información a la siguiente capa), algunas **normalizan la salida a un rango**, y **afectan cómo se actualizan los pesos** —porque la retropropagación usa su derivada—.

| Función | Fórmula | Rango | Nota |
|---|---|---|---|
| **Sigmoide (logística)** | `1 / (1 + e^(−x))` | (0, 1) | salida legible como probabilidad; **satura** en los extremos |
| **Tanh** | `(e^z − e^(−z)) / (e^z + e^(−z))` | (−1, 1) | igual forma pero **centrada en 0**; converge más rápido; también satura |
| **ReLU** | `0 si x < 0; x si x ≥ 0` | [0, ∞) | derivada 1 en el lado positivo: **no satura**; barata; default en capas ocultas |

> **Errata del material:** la slide escribe la sigmoide como `1/(1 − e^(−x))`. Es un error tipográfico: va **más**. Con el signo menos la función se indefine en `x = 0` y no coincide con la curva dibujada en la misma slide.

**Saturación y gradiente desvaneciente.** En los extremos, sigmoide y tanh son casi planas: su derivada tiende a 0. Como la retropropagación **multiplica** derivadas capa por capa, en redes profundas el gradiente se apaga y las primeras capas dejan de aprender. ReLU no tiene ese problema por el lado positivo; a cambio, una neurona cuya entrada queda siempre negativa tiene gradiente 0 y muere (*dying ReLU*).

> **Nota (verificado en notebook):** la derivada en `x = 6` vale **0,0025** para la sigmoide y **0,00002** para tanh, contra **1** para ReLU: la saturación es medible, no una figura retórica. Y sobre el XOR, un `MLPClassifier` con `activation='identity'` se queda en **50%** de exactitud —no lo resuelve—, mientras que con `tanh` o `relu` llega a **100%**. La no linealidad es lo único que cambia entre esos tres casos.

**Qué usar:** ReLU en capas ocultas por defecto; **sigmoide** en la salida binaria; **softmax** en la salida multiclase; **ninguna** (identidad) en la salida de regresión. En `scikit-learn`, `activation` aplica **solo a las capas ocultas** —la de salida la elige el estimador según el problema— y acepta `'relu'` (default), `'tanh'`, `'logistic'`, `'identity'`.

### 33. Grafos y capa densa

**Apuntes.**

Un **grafo** describe cómo están conectadas las unidades: **nodos** (neuronas) y **aristas** (conexiones, cada una con su peso). En una **capa densa** o **totalmente conectada**, **cada neurona está conectada a todas las neuronas de la capa anterior**.

Consecuencia práctica: entre una capa de `n` neuronas y una densa de `m` hay `n × m` pesos más `m` sesgos. Por eso los parámetros crecen como un **producto**, no como una suma, al agrandar la red.

El diagrama del curso rotula cada neurona oculta como **`Σf`**, y el rótulo es literal: **suma ponderada** (`Σ`) seguida de la **función de activación** (`f`). Es decir, cada nodo es un perceptrón (§27); la red es muchos de ellos conectados. En `scikit-learn`, `hidden_layer_sizes=(100, 150)` describe exactamente ese grafo, y los pesos quedan en `mlp.coefs_` y los sesgos en `mlp.intercepts_`.

### 34. Diseño de la arquitectura de la red

**Apuntes.**

Elegir el **número de capas** y de **neuronas** afecta directamente la **capacidad de aprendizaje**, el **poder de generalización** y el **rendimiento computacional**.

| Decisión | Si se pasa | Si se queda corto |
|---|---|---|
| **Capas** | más complejidad y más tiempo de entrenamiento | no aprende representaciones complejas |
| **Neuronas** | **sobreajuste**: memoriza el train | **capacidad insuficiente** para la complejidad de los datos |

No hay fórmula: la arquitectura es un **hiperparámetro** y se busca comparando en validación, nunca en train.

> **Nota (verificado en notebook, Iris con 2 atributos, `relu`):** `(2,)` da 46,7% de train y 56,7% de test con 15 parámetros —**underfitting** de manual: no ajusta ni el entrenamiento—; `(100,)` llega a 100% de test con 603 parámetros; y `(100, 100)`, con **10.703** parámetros (~17 veces más), **no mejora nada**: baja a 96,7%. La lectura honesta es "capacidad de sobra sin beneficio" más que sobreajuste probado, porque con **30 muestras de test** un acierto vale 3,3 puntos y las diferencias chicas no son concluyentes.

> **Nota (verificado en notebook):** escalar con `StandardScaler` **bajó la pérdida final en las tres activaciones** (tanh 0,243 → 0,161; relu 0,197 → 0,177), pero el efecto sobre la exactitud fue **mixto**: `logistic` anduvo mejor *sin* escalar (100% contra 90%). Ninguna de las seis corridas convergió dentro de `max_iter=300`. El escalado es buena práctica por lo que le hace al entrenamiento, no porque garantice mejor exactitud en un dataset chico. Método práctico: arrancar simple (una capa oculta), agrandar solo si el error de *entrenamiento* sigue alto, y si el error de train es bajo pero el de test alto, achicar o regularizar (`alpha`) antes que agregar capas. Verificar lo que realmente quedó entrenado con `mlp.n_layers_`, `mlp.coefs_` y `mlp.loss_curve_`.

### 35. Funciones de pérdida

**Apuntes.**

La **función de pérdida** (o **función de costo**) cuantifica la **discrepancia entre lo que el modelo predijo y los valores reales**, produciendo **un único valor**. Ese valor es la **señal para ajustar pesos y sesgos**: es lo que la retropropagación deriva. En `scikit-learn` es la curva `mlp.loss_curve_`. La elección **depende del tipo de problema**.

**Regresión.**

```
MAE = (1/n) · Σ |yᵢ − ŷᵢ|          MSE = (1/n) · Σ (yᵢ − ŷᵢ)²
```

El MSE eleva al cuadrado: castiga mucho más los errores grandes y es **sensible a outliers**; el MAE es más **robusto**. A cambio, el MSE es derivable en todo su dominio, lo que lo hace más cómodo para el descenso de gradiente. `MLPRegressor` minimiza MSE.

**Clasificación.**

```
BCE:  L = −(1/N) · Σᵢ ( yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) )
CCE:  L = −(1/N) · Σⱼ Σᵢ  yⱼᵢ·log(ŷⱼᵢ)
```

La **entropía binaria cruzada (BCE)** se usa en clasificación **binaria**; la **entropía cruzada categórica (CCE)**, en **multiclase**. Ambas miden la diferencia entre la distribución verdadera y la predicha. Con `y` en formato *one-hot*, en la CCE solo sobrevive el término de la clase correcta: la pérdida es `−log` de la probabilidad que el modelo le asignó a esa clase. `MLPClassifier` **no expone el parámetro**: usa log-loss siempre, que es BCE en el caso binario y CCE en el multiclase.

**Pérdida ≠ métrica de evaluación.** La pérdida es lo que se **minimiza** durante el entrenamiento y debe ser derivable; la métrica (exactitud, R², RMSE) es lo que se **reporta**. A veces coinciden (MSE) y a veces no: nadie entrena minimizando exactitud, porque no es derivable.

### 36. Optimización y descenso de gradiente

**Apuntes.**

**Optimizar** es ajustar pesos y sesgos para **minimizar la función de pérdida** (§35). El método es el **descenso de gradiente**: se calcula el **gradiente** —la derivada de la función de error respecto de *todos* los parámetros de la red— y se avanza en sentido contrario.

**La regla de actualización**, que es toda la idea en una línea:

```
θ = θ − η · ∇θ J(θ)
```

donde `θ` son los parámetros (pesos y sesgos), `η` la **tasa de aprendizaje** y `∇θ J(θ)` el gradiente de la pérdida respecto de `θ`. El **signo menos** es el punto: el gradiente apunta hacia donde la pérdida *crece*, así que minimizar es moverse al revés. Visualmente, la pérdida es una superficie con forma de cuenco y entrenar es bajar hasta el fondo.

**La tasa de aprendizaje es el hiperparámetro más sensible.**

| `η` | Qué pasa |
|---|---|
| **Muy chica** | converge, pero lento: puede agotar `max_iter` sin llegar |
| **Adecuada** | pocos pasos, convergencia estable |
| **Muy grande** | salta de un lado al otro del valle y **diverge**: la pérdida oscila o explota |

**Tres modos de descenso**, según cada cuánto se actualizan los pesos:

| Modo | Actualiza | Característica |
|---|---|---|
| **Estocástico (SGD)** | cada vez que se evalúa **una muestra** | muy frecuente y ruidoso |
| **En lotes (batch)** | al terminar **una época** (todo el train) | estable, pero pocas actualizaciones y caro en memoria |
| **En mini-lotes** | al terminar cada **minilote**, con la media del gradiente del lote | el compromiso: es lo que se usa en la práctica |

> **Ojo con el nombre:** `solver='sgd'` en `scikit-learn` trabaja en **mini-lotes** (`batch_size`), no de a una muestra. El nombre es histórico.

**Dónde encaja:** la **regla delta** del perceptrón (§28) es este mismo mecanismo en su versión mínima; **backpropagation** (§31) es el algoritmo que calcula ese gradiente para todas las capas aplicando la regla de la cadena hacia atrás; y la **saturación** de las activaciones (§32) es lo que lo rompe, porque multiplica el gradiente por números casi nulos.

### 37. Regularización

**Apuntes.**

La **regularización** es el conjunto de técnicas para **prevenir el sobreajuste** y mejorar la **generalización**, controlando la complejidad del modelo y evitando que los pesos se vuelvan **demasiado grandes o especializados** en el train. Dos advertencias de la propia slide: **no hay una técnica universalmente superior**, y **aumenta el tiempo de entrenamiento**.

| Técnica | Penalización | Efecto sobre los pesos |
|---|---|---|
| **L1** | `L(X, w) + λ · Σ \|wᵢ\|` — suma de **valores absolutos** | lleva pesos **a cero**: selecciona variables, red rala |
| **L2** | `L(X, w) + λ · Σ wᵢ²` — suma de **cuadrados** | los **encoge** hacia cero sin anularlos |
| **Dropout** | — | **apaga neuronas al azar** durante el entrenamiento |

Es la misma distinción L1/L2 que Lasso y Ridge en regresión lineal (§21). **Dropout** es de otra naturaleza: no toca la pérdida sino la arquitectura durante el entrenamiento, y al apagar neuronas al azar impide que la red dependa de una neurona en particular.

| Ventajas | Desventajas |
|---|---|
| previene el sobreajuste | más costo computacional |
| controla la complejidad | suma hiperparámetros que hay que elegir |
| puede acelerar la convergencia | posible pérdida parcial de información |

**En `scikit-learn` solo hay L2**, vía `alpha` (la `λ` de la fórmula), con default `0.0001`. **No hay L1 ni dropout** para redes: eso requiere Keras o PyTorch. Sí está disponible la **parada temprana** (`early_stopping=True`), que es regularización de hecho: cortar cuando la validación deja de mejorar evita seguir ajustando ruido.

**Cómo se busca `alpha`:** con validación cruzada, graficando exactitud de **train y de validación** contra `alpha`. La señal de sobreajuste es la **brecha** entre las dos curvas; el buen `alpha` la cierra sin hundir las dos.

> **Nota (verificado en el recurso de clase):** un barrido de `alpha` sobre un problema demasiado fácil **no muestra nada**. En `OD_RN1_ESP_M03_S10`, cinco valores de `alpha` que cubren cinco órdenes de magnitud (de 0,00001 a 1,0) dan **todos la misma exactitud, 0,97**, y los tres solvers también. Con Iris de 2 atributos y 30 muestras de test, la exactitud solo puede valer 29/30 o 30/30: no hay resolución para distinguir configuraciones. Para ver el efecto de la regularización hace falta un problema que efectivamente sobreajuste.

### Referencia técnica — Módulo 06

**Cuándo aplica un perceptrón simple:** problema de **clasificación binaria** con clases **linealmente separables**; sirve como bloque base para entender MLP, pero en la práctica rara vez se usa solo (no resuelve XOR ni problemas no lineales).

**Hiperparámetros clave:**

| Hiperparámetro | Rol | Notas |
|---|---|---|
| Tasa de aprendizaje (`eta0` en sklearn) | Magnitud del ajuste de pesos en cada actualización | Muy alta → oscila sin converger; muy baja → converge lento |
| Inicialización de pesos/bias | Punto de partida del entrenamiento | En el ejemplo manual se parte de valores pequeños no nulos (`w1=0.381`, `w2=0.245`, `θ=0.196`); en la práctica suele inicializarse en 0 o con valores aleatorios chicos |
| Número máximo de iteraciones (`max_iter`) | Corte si no converge | Relevante cuando los datos no son linealmente separables (§29) |

```python
from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # x1, x2
y = np.array([0, 0, 0, 1])                        # x1 AND x2

clf = Perceptron(max_iter=1000, eta0=0.045, random_state=42)
clf.fit(X, y)

clf.predict(X)        # array([0, 0, 0, 1]) → aprende la compuerta AND
clf.coef_, clf.intercept_   # pesos (w1, w2) y bias (equivalente a -θ)
```

**Flujo completo sobre un dataset real (Iris, setosa vs versicolor):**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
mask = iris.target < 2                       # solo clases 0 y 1 (binario)
X = iris.data[mask][:, [0, 2]]               # sepal length, petal length
y = iris.target[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = Perceptron(max_iter=10, eta0=0.05, random_state=42).fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)               # 1.00 (clases linealmente separables)
clf.n_iter_                                  # 7 → convergió antes de max_iter
```

**Puntos de atención del flujo:**

- `random_state` en el split **y** en el modelo: sin eso cada corrida da pesos y métricas distintos y los resultados no son comparables.
- `stratify=y` mantiene la proporción de clases en train y test; importa con particiones chicas.
- `accuracy_score(y_verdadero, y_predicho)` — ese orden. Invertirlo no cambia la exactitud pero sí transpone la matriz de confusión.
- `n_iter_ < max_iter` significa que **convergió**: dejó de haber errores y el algoritmo cortó solo.
- **Frontera de decisión** en 2D: de `w1·x1 + w2·x2 + b = 0` se despeja `x2 = -(w1/w2)·x1 - b/w2`. El perceptrón se queda con la primera recta sin errores, **no** con la de mayor margen (esa es la diferencia con SVM).
- Comparar magnitudes de pesos solo tiene sentido si los atributos están en escalas parecidas; si no, estandarizar antes (`StandardScaler`).
- `decision_function(X)` devuelve `z` **antes** del escalón: el signo da la clase y la magnitud, la distancia relativa a la frontera.

#### Perceptrón multicapa — `MLPClassifier` / `MLPRegressor`

**Cuándo aplica:** relaciones **no lineales** entre atributos y salida, con suficientes datos. Si el problema es linealmente separable o hay pocas muestras, un modelo lineal es preferible: más barato, interpretable y con menos riesgo de sobreajuste.

**Diferencias entre los dos estimadores:**

| | `MLPClassifier` | `MLPRegressor` |
|---|---|---|
| Predice | clase discreta | valor continuo |
| Capa de salida | una neurona por clase | una sola, sin activación |
| Pérdida | log-loss | error cuadrático medio |
| Métricas | exactitud, precision, recall, F1 | MSE, RMSE, MAE, R² |

Todo lo demás —capas ocultas, activación no lineal, backpropagation— es idéntico.

**Hiperparámetros clave (comunes a ambos):**

| Hiperparámetro | Rol | Notas |
|---|---|---|
| `hidden_layer_sizes` | arquitectura, ej. `(100, 150)` = dos capas ocultas | **a buscar, no a maximizar**: hay un mínimo por debajo del cual subajusta, y a partir de cierto punto solo suma costo y sobreajuste |
| `activation` | no linealidad de las ocultas: `relu` (default), `tanh`, `logistic`, `identity` | con `identity` la red colapsa a un modelo lineal — las fronteras vuelven a ser rectas |
| `alpha` | regularización L2 | subirlo suaviza las fronteras y contiene el sobreajuste |
| `max_iter` | tope de épocas | si salta `ConvergenceWarning`, el modelo sirve pero podría mejorar con más épocas |
| `early_stopping` | corta cuando deja de mejorar sobre un split de validación | evita gastar épocas de más |

**Diagnóstico del entrenamiento** (no omitirlo): `n_iter_` dice si realmente convergió o si se cortó por `max_iter`; `loss_curve_` debe **bajar y aplanarse** (estancada alto = poca capacidad; cayendo aún al final = faltaron épocas; oscilando = tasa de aprendizaje alta). Ojo: esa curva es la pérdida **de entrenamiento** — que baje a cero puede ser justamente la señal de que memorizó.

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# El escalado NO es opcional: fit_transform en train, solo transform en test
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)          # nunca fit sobre test -> data leakage

# Clasificación
clf = MLPClassifier(hidden_layer_sizes=(20,), activation="relu",
                    max_iter=2000, random_state=42).fit(X_train_s, y_train)
clf.n_iter_, clf.loss_curve_[-1]

# Regresión
reg = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=0.001,
                   max_iter=1000, random_state=42).fit(X_train_s, y_train)
```

**Evaluación en regresión:** el **RMSE** es la métrica interpretable, porque vuelve a las unidades del objetivo; el **MSE** sirve para comparar modelos entre sí; el **MAE** es más robusto a valores extremos; el **R²** dice qué proporción de la varianza se explica (0 = equivale a predecir la media, negativo = peor que la media).

**Una métrica sola no alcanza:** compararla siempre contra un **baseline** (predecir la media, y un modelo lineal). Si el MLP no le gana a una regresión lineal, su complejidad no se justifica.

> **Nota (verificado en notebook, California Housing):** predecir la media da R² = 0; la regresión lineal, 0,576; el `MLPRegressor`, **0,799** (RMSE 0,513 ≈ 51.000 dólares). El MLP se justifica porque las relaciones del dataset no son lineales.

**Antes de culpar al modelo, mirar los datos.** En California Housing el objetivo está **truncado** artificialmente en 5.00001 (500.000 dólares): 992 distritos —casi el 5% del dataset— comparten ese valor. Eso pone un techo al rendimiento alcanzable y aparece como una banda vertical en el gráfico de reales vs predicciones. No se descubre mirando métricas agregadas; se descubre mirando los datos. Los gráficos de **reales vs predicciones** y de **residuos** muestran *dónde* falla el modelo, cosa que un número agregado nunca dice.

#### Optimización: solvers y tasa de aprendizaje

| `solver` | Qué es | Cuándo conviene |
|---|---|---|
| `adam` (default) | SGD con tasa de aprendizaje **adaptativa por parámetro** | opción por defecto en datasets chicos y medianos |
| `sgd` | descenso por gradiente en mini-lotes, con `momentum` | cuando se quiere control fino; **requiere ajustar `learning_rate_init`** |
| `lbfgs` | método cuasi-Newton (segundo orden), sin mini-lotes | **datasets chicos**: converge en pocas iteraciones y suele ganar; no escala a datos grandes |

**Parámetros que solo aplican a `sgd` / `adam`** (con `lbfgs` se ignoran):

| Parámetro | Rol |
|---|---|
| `learning_rate_init` | la `η` de la fórmula: tamaño del paso inicial |
| `learning_rate` | `'constant'`, `'invscaling'` o `'adaptive'` — cómo evoluciona `η` (solo `sgd`) |
| `momentum` | inercia que acelera en la dirección sostenida y amortigua el zigzag (solo `sgd`) |
| `batch_size` | tamaño del mini-lote; default `min(200, n_muestras)` |
| `n_iter_no_change` + `tol` | parada temprana: cuántas épocas sin mejorar mayor a `tol` se toleran |

```python
MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', learning_rate='adaptive',
              learning_rate_init=0.01, momentum=0.9, batch_size=50,
              n_iter_no_change=20, tol=1e-4, random_state=42)
```

#### Búsqueda de hiperparámetros

```python
from sklearn.model_selection import GridSearchCV

grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
gs = GridSearchCV(MLPClassifier(max_iter=1000), grid, cv=5, n_jobs=-1)
gs.fit(X_train_s, y_train)
gs.best_params_, gs.best_score_
```

**Tres trampas de esta celda:**

- **Sin `random_state`, el resultado no es reproducible.** Es el error más costoso, porque no avisa: el estimador devuelve una combinación ganadora con toda seriedad y a la corrida siguiente devuelve otra.
- **`max_iter` bajo invalida la comparación.** Si los candidatos no convergen (`ConvergenceWarning`), el "mejor" resultado dice cuál **arranca más rápido**, no cuál es mejor. Darle margen a `max_iter` o el ranking es ruido.
- **La grilla explota.** El ejemplo son 3 × 2 × 2 × 2 × 2 = **48 combinaciones**, por `cv=5` = **240 entrenamientos**. Para grillas grandes, `RandomizedSearchCV` cubre más espacio con el mismo presupuesto.
- **Escalar dentro del CV, no antes.** Si se estandariza sobre todo el train antes de partirlo, cada pliegue de validación ve estadísticas calculadas con sus propios datos: eso es **fuga de información** y el score sale optimista. La forma correcta es un `Pipeline(StandardScaler(), MLPClassifier())` como estimador del grid.

#### Diagnóstico: qué mirar cuando el resultado no cierra

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `ConvergenceWarning`, `n_iter_ == max_iter` | no terminó de entrenar | subir `max_iter`; revisar escalado |
| `loss_curve_` estancada alto | capacidad insuficiente o `η` muy chica | más neuronas/capas; subir `learning_rate_init` |
| `loss_curve_` oscilando | `η` demasiado grande | bajar `learning_rate_init`; `learning_rate='adaptive'` |
| Pérdida de train cae a ~0 y el test empeora | **sobreajuste** | subir `alpha`, achicar la red, `early_stopping=True` |
| Train y test igual de malos | **subajuste** | agrandar la red, bajar `alpha`, más atributos |
| Resultados que cambian en cada corrida | falta `random_state` | fijarlo en el split **y** en el estimador |
| Métricas raras con clases desbalanceadas | la exactitud engaña | mirar matriz de confusión, precision/recall, F1 |

> **Nota (verificado, recurso `OD_RN1_ESP_M03_S10`):** el `GridSearchCV` de ese notebook usa `MLPClassifier(max_iter=100)` sin `random_state`. Ejecutado **cuatro veces seguidas** devuelve **cuatro combinaciones ganadoras distintas**, con scores entre 0,950 y 0,967, y ninguna coincide con la que quedó guardada en el notebook. Lo único estable es `solver='adam'`. Cuando los candidatos empatan dentro del ruido, la búsqueda no está eligiendo hiperparámetros: está sorteando semillas.

**Antes de creerle a una comparación**, revisar dos cosas: que el conjunto de test tenga **tamaño suficiente** —con 30 muestras, un acierto vale 3,3 puntos y casi nada es concluyente— y que todas las configuraciones hayan **convergido**. Sin eso, la tabla de resultados mide ruido.

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
| **Función de activación** | No linealidad que aplica cada neurona tras la suma ponderada (ReLU, tanh, sigmoide). |
| **Saturación** | Zona plana de una activación donde su derivada tiende a 0 y el aprendizaje se frena. |
| **Gradiente desvaneciente** | Gradiente que se apaga al propagarse hacia atrás por multiplicar derivadas chicas. |
| **Capa densa** | Capa totalmente conectada: cada neurona se conecta a todas las de la capa anterior. |
| **Función de pérdida / costo** | Valor único que cuantifica el error de las predicciones; es lo que se minimiza. |
| **BCE / CCE** | Entropía cruzada binaria / categórica; pérdidas de clasificación. |
| **Descenso de gradiente** | `θ = θ − η·∇θJ(θ)`: mover los parámetros en contra del gradiente. |
| **Tasa de aprendizaje (η)** | Tamaño del paso en cada actualización; muy alta diverge, muy baja va lenta. |
| **Época** | Una pasada completa por todo el conjunto de entrenamiento. |
| **Mini-lote (batch)** | Subconjunto de muestras tras el cual se actualizan los pesos. |
| **Solver** | Algoritmo que optimiza los pesos: `adam`, `sgd`, `lbfgs`. |
| **Momentum** | Inercia que acumula dirección entre pasos para acelerar y estabilizar el SGD. |
| **Dropout** | Regularización que apaga neuronas al azar durante el entrenamiento. |
| **Parada temprana** | Cortar el entrenamiento cuando la validación deja de mejorar. |
| **alpha (sklearn)** | Coeficiente de regularización L2 en `MLPClassifier`/`MLPRegressor`. |
