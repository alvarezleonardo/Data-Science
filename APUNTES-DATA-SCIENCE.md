# Apuntes Data Science — Digital House

> Apuntes consolidados y mejorados a partir del material del curso (slides en PDF).
> Cada PDF original tiene además su conversión 1:1 en un `.md` al lado del archivo.
> Donde el material de origen tenía errores o ambigüedades, se corrigen y se marcan con **Nota**.

## Índice

- [Módulo 1 — Introducción a Machine Learning](#módulo-1--introducción-a-machine-learning)
  - [1. ¿Qué es Machine Learning?](#1-qué-es-machine-learning)
  - [2. Datasets, features y labels](#2-datasets-features-y-labels)
  - [3. Tipos de aprendizaje](#3-tipos-de-aprendizaje)
  - [4. Ciclo de vida de un proyecto de ML](#4-ciclo-de-vida-de-un-proyecto-de-ml)
  - [5. Preparación de datos y EDA](#5-preparación-de-datos-y-eda)
  - [6. Clasificadores lineales](#6-clasificadores-lineales)
  - [7. Árboles de decisión](#7-árboles-de-decisión)
  - [8. Métricas de clasificación](#8-métricas-de-clasificación)
  - [9. Aprendizaje no supervisado](#9-aprendizaje-no-supervisado)
  - [10. Evaluación de modelos](#10-evaluación-de-modelos)
  - [11. XGBoost y ensambles](#11-xgboost-y-ensambles)
- [Módulo 2 — Modelado avanzado](#módulo-2--modelado-avanzado)
  - [12. Regresión lineal](#12-regresión-lineal)
  - [13. Métricas de regresión](#13-métricas-de-regresión)
  - [14. Inferencia sobre los coeficientes](#14-inferencia-sobre-los-coeficientes)
- [Módulo 3 — Desafío Profesional](#módulo-3--desafío-profesional)
- [Glosario rápido](#glosario-rápido)

---

## Módulo 1 — Introducción a Machine Learning

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

### 9. Aprendizaje no supervisado

Sin etiquetas: el objetivo es **descubrir patrones/estructura**.

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

## Módulo 2 — Modelado avanzado

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

### 14. Inferencia sobre los coeficientes

**Test de significación individual** para `βⱼ`:

- **H₀:** `βⱼ = 0` (no hay relación entre `Xⱼ` e `Y`).
- **H₁:** `βⱼ ≠ 0` (hay relación).

Si `β₁ = 0`, el modelo se reduce a `Y = β₀ + ε` y `X` no aporta. Se evalúa con el **p-value**:

- **p-value chico** (< 0,05 típicamente) → se rechaza H₀: hay evidencia de relación.
- También se usan **intervalos de confianza** para los coeficientes.

---

## Módulo 3 — Desafío Profesional

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

## Glosario rápido

| Término | Definición |
|---------|-----------|
| **Feature** | Variable de entrada (`X`). |
| **Label / Target** | Variable a predecir (`y`). |
| **Hiperparámetro** | Configuración fijada antes de entrenar (ej. `max_depth`). |
| **Overfitting** | Memoriza el train, generaliza mal. |
| **Underfitting** | Demasiado simple, no captura el patrón. |
| **Hiperplano** | Frontera de decisión lineal. |
| **Impureza (Gini/Entropía)** | Mezcla de clases en un nodo; 0 = puro. |
| **Ensamble** | Combinación de varios modelos (bagging/boosting). |
| **OLS** | Mínimos cuadrados ordinarios. |
| **RSS / TSS / ESS** | Suma de cuadrados Residual / Total / Explicada. |
