# Manual de Data Science — Digital House

> Manual de estudio del programa, ordenado **de lo más básico a lo más complejo**: cada parte se apoya en la anterior. No sigue el orden de la cursada — para eso está la [tabla de equivalencia](#equivalencia-con-los-módulos-del-programa) más abajo.
>
> Cada tema combina **dos criterios**: los **apuntes** (la teoría explicada para entender y estudiar) y, en la Parte VIII, la **referencia técnica** (cuándo aplicar cada técnica, hiperparámetros, cómo se evalúa y snippets de `scikit-learn`).
>
> Cada PDF del curso tiene además su conversión 1:1 en un `.md` al lado del archivo, en la carpeta del módulo. Donde el material de origen tenía errores, se corrigen y se marcan con **Nota**. Las notas que dicen **verificado en notebook** contienen resultados medidos, no citados.

## Índice

**[Parte I — Fundamentos](#parte-i--fundamentos)**
[1. ¿Qué es Machine Learning?](#1-qué-es-machine-learning) · [2. Datasets, features y labels](#2-datasets-features-y-labels) · [3. Tipos de aprendizaje](#3-tipos-de-aprendizaje) · [4. El ciclo de vida de un proyecto de ML](#4-el-ciclo-de-vida-de-un-proyecto-de-ml) · [5. Preparación de datos y EDA](#5-preparación-de-datos-y-eda)

**[Parte II — Cómo se evalúa un modelo](#parte-ii--cómo-se-evalúa-un-modelo)**
[6. Train/test y validación cruzada](#6-traintest-y-validación-cruzada) · [7. Sobreajuste, subajuste y sesgo-varianza](#7-sobreajuste-subajuste-y-el-compromiso-sesgo-varianza) · [8. Métricas de clasificación](#8-métricas-de-clasificación) · [9. Métricas de regresión](#9-métricas-de-regresión)

**[Parte III — Modelos lineales](#parte-iii--modelos-lineales)**
[10. Clasificadores lineales](#10-clasificadores-lineales) · [11. Regresión lineal](#11-regresión-lineal) · [12. Inferencia sobre los coeficientes](#12-inferencia-sobre-los-coeficientes)

**[Parte IV — Modelos no lineales](#parte-iv--modelos-no-lineales)**
[13. Árboles de decisión](#13-árboles-de-decisión) · [14. Ensambles y XGBoost](#14-ensambles-y-xgboost)

**[Parte V — Complejidad del modelo y selección](#parte-v--complejidad-del-modelo-y-selección)**
[15. Regularización: Ridge, Lasso y Elastic Net](#15-regularización-ridge-lasso-y-elastic-net) · [16. Selección de variables](#16-selección-de-variables) · [17. Criterios de selección de modelos (AIC y BIC)](#17-criterios-de-selección-de-modelos-aic-y-bic)

**[Parte VI — Aprendizaje no supervisado](#parte-vi--aprendizaje-no-supervisado)**
[18. Clustering con K-means](#18-clustering-con-k-means) · [19. Clustering jerárquico](#19-clustering-jerárquico) · [20. DBSCAN](#20-dbscan-clustering-por-densidad) · [21. Evaluación de clusters](#21-evaluación-de-clusters) · [22. La maldición de la dimensión](#22-la-maldición-de-la-dimensión) · [23. Reducción de dimensionalidad](#23-reducción-de-dimensionalidad)

**[Parte VII — Redes neuronales](#parte-vii--redes-neuronales)**
[24. Fundamentos biológicos](#24-fundamentos-biológicos) · [25. Historia](#25-historia-de-las-redes-neuronales) · [26. El perceptrón](#26-el-perceptrón-estructura-y-fórmulas) · [27. Entrenamiento: la compuerta AND](#27-entrenamiento-del-perceptrón-la-compuerta-and) · [28. Limitaciones](#28-limitaciones-del-perceptrón) · [29. Implementación con scikit-learn](#29-implementación-con-scikit-learn) · [30. El perceptrón multicapa](#30-el-perceptrón-multicapa-mlp) · [31. Grafos y capa densa](#31-grafos-y-capa-densa) · [32. Funciones de activación](#32-funciones-de-activación) · [33. Diseño de la arquitectura](#33-diseño-de-la-arquitectura-de-la-red) · [34. Funciones de pérdida](#34-funciones-de-pérdida) · [35. Optimización y descenso de gradiente](#35-optimización-y-descenso-de-gradiente) · [36. Regularización en redes](#36-regularización-en-redes-neuronales) · [37. Backpropagation](#37-backpropagation) · [38. Persistencia de modelos](#38-persistencia-de-modelos)

**[Parte VIII — Referencia técnica](#parte-viii--referencia-técnica)** · **[Desafíos profesionales](#desafíos-profesionales)** · **[Glosario](#glosario-rápido)**

## Equivalencia con los módulos del programa

El manual reordena el contenido por dificultad. Esta tabla mapea cada módulo de la cursada a los capítulos donde quedó su material.

| Módulo del programa | Capítulos |
|---|---|
| **01** — Introducción a Machine Learning | 1-5, 6, 8, 10, 13, 14 |
| **02** — Desafío Profesional (Etapa 1) | [Desafíos profesionales](#desafíos-profesionales) |
| **03** — Modelado avanzado en ML | 9, 11, 12 |
| **04** — Aprendizaje no supervisado | 15, 16, 17, 18-23 |
| **05** — Desafío Profesional (Etapa 2) | [Desafíos profesionales](#desafíos-profesionales) |
| **06** — Fundamentos de redes neuronales | 24-38 |
| **07** — Fundamentos de deep learning | pendiente |
| **08** — Gestión de proyectos de IA | pendiente |

> **Capítulos que no vienen de una slide.** El 7 (sobreajuste y sesgo-varianza) es una ampliación propia: el material del curso lo da por sabido. Los capítulos 5, 6, 10 y 14 están ampliados bastante más allá de lo que cubren las slides.

---

## Parte I — Fundamentos

Antes de tocar un solo algoritmo hace falta el vocabulario mínimo: qué es aprender de datos, cómo se llaman las piezas de un dataset, qué familias de problemas existen y qué pasos sigue un proyecto real de punta a punta. Esta parte no todavía no entrena ningún modelo serio — arma el mapa sobre el que se van a ubicar todas las técnicas de las partes siguientes (clasificadores lineales, árboles, ensambles, clustering, reducción de dimensionalidad, redes neuronales). Cierra con el tema que más tiempo real insume en cualquier proyecto y menos espacio ocupa en los cursos: preparar los datos correctamente, evitando el error silencioso más común del oficio, el data leakage.

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

**Panorama del no supervisado.** Sin etiquetas, el objetivo deja de ser predecir y pasa a ser **descubrir patrones/estructura** en los datos:

- **Clustering:** agrupar observaciones similares y separar las distintas. Usos: segmentación de mercado, detección de anomalías. Algoritmo típico: **K-Means** (cap. 18).
- **Reducción de dimensionalidad:** comprimir muchas features en pocas conservando la información relevante; útil para visualizar y reducir ruido. Algoritmo típico: **PCA** (cap. 23).

El desarrollo completo de clustering, reducción de dimensionalidad y selección de variables está más adelante en el manual, una vez cubiertas evaluación y regularización (caps. 15 a 23 según el temario).

### 4. El ciclo de vida de un proyecto de ML

1. **Definición del problema** — entender contexto y negocio, definir la pregunta, objetivos y métricas de éxito, variables de entrada y target.
2. **Preparación de datos** — recolección, EDA, limpieza, selección/ingeniería de features (cap. 5).
3. **Selección y entrenamiento** — elegir algoritmo según datos y requisitos (clasificadores lineales, cap. 10; árboles, cap. 13; ensambles, cap. 14); entrenar y ajustar hiperparámetros.
4. **Evaluación** — medir con métricas acordes al problema: clasificación (cap. 8) vs regresión (cap. 9), usando train/test y validación cruzada (cap. 6).
5. **Optimización y refinamiento** — tuning de hiperparámetros, feature engineering, regularización (cap. 15) para corregir el sobreajuste (cap. 7).
6. **Implementación en producción** — despliegue del modelo.
7. **Monitoreo y mantenimiento** — vigilar *drift* y reentrenar.

> **Nota:** las slides repetían el mismo texto en "Evaluación" y "Optimización". La optimización es iterativa: tras evaluar se ajustan hiperparámetros y features y se vuelve a evaluar.

```mermaid
flowchart LR
    A[Definir problema] --> B[Preparar datos]
    B --> C[Entrenar modelo]
    C --> D[Evaluar]
    D --> E[Optimizar]
    E --> C
    D --> F[Producción]
    F --> G[Monitoreo]
    G --> B
```

### 5. Preparación de datos y EDA

El **EDA (Análisis Exploratorio de Datos)** combina estadística descriptiva y visualizaciones para entender distribución, relaciones, outliers y valores faltantes antes de entrenar nada. La **selección de features** —quedarse con las variables más relevantes para el target— reduce ruido y overfitting; se retoma en profundidad más adelante (selección de variables). Lo que sigue detalla las tareas concretas que arma un pipeline de preparación de datos: tipos de variable, faltantes, codificación, escalado, outliers y el error más costoso de esta etapa, el data leakage.

**Tipos de variables.**

| Tipo | Ejemplos | Tratamiento típico |
|------|----------|---------------------|
| **Numéricas continuas** | precio, edad, temperatura | escalado, tratamiento de outliers |
| **Numéricas discretas** | cantidad de hijos, nº de habitaciones | igual que continuas, a veces se tratan como categóricas si hay pocos valores |
| **Categóricas nominales** | color, ciudad, tipo de producto | one-hot encoding |
| **Categóricas ordinales** | nivel educativo, talle (S/M/L) | encoding ordinal (respeta el orden) |

**Valores faltantes.** Antes de imputar hay que entender por qué faltan (no siempre es al azar). Estrategias comunes:

| Estrategia | Cuándo usarla |
|------------|----------------|
| Eliminar filas | Pocos faltantes, dataset grande |
| Eliminar columna | La columna tiene demasiados faltantes (ej. > 60-70%) |
| Imputar con media/mediana | Numéricas; mediana si hay outliers |
| Imputar con moda | Categóricas |
| Imputar con un modelo (KNN, regresión) | Cuando el patrón de faltantes es informativo y hay volumen suficiente |
| Crear una columna indicadora de "faltante" | Cuando el hecho de faltar puede ser predictivo en sí mismo |

**Codificación de categóricas: one-hot vs ordinal.**

- **One-hot encoding:** crea una columna binaria por categoría. Usar con **nominales** (sin orden), porque no le impone al modelo una relación de magnitud que no existe.
- **Encoding ordinal:** asigna un número que respeta el orden (ej. `bajo=0, medio=1, alto=2`). Usar solo con **ordinales**; aplicarlo a una nominal introduce un orden falso que el modelo puede interpretar como relación numérica.

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

OneHotEncoder(handle_unknown="ignore").fit_transform(X[["ciudad"]])          # nominal
OrdinalEncoder(categories=[["bajo", "medio", "alto"]]).fit_transform(X[["nivel"]])  # ordinal
```

**Escalado: `StandardScaler` vs `MinMaxScaler`.**

| | `StandardScaler` | `MinMaxScaler` |
|---|---|---|
| Transformación | media 0, desvío 1 (`z = (x-μ)/σ`) | rango fijo, típicamente `[0,1]` |
| Sensible a outliers | Menos | Sí (un outlier extremo comprime todo el resto) |
| Cuándo usarlo | Métodos basados en distancias o varianza: K-means, PCA, regresión regularizada (Ridge/Lasso), redes neuronales | Cuando se necesita un rango acotado explícito (ej. entrada de ciertas redes, imágenes) |

> **Nota:** modelos basados en árboles (árboles de decisión, Random Forest, XGBoost) son invariantes a escala — dividen por umbrales, no por distancia — así que no necesitan escalado.

**Outliers.** Se detectan con boxplot/IQR (`Q1 - 1.5·IQR`, `Q3 + 1.5·IQR`) o con z-score (`|z| > 3`). Tratamiento según el caso: eliminarlos si son errores de carga, capearlos (winsorizing), transformarlos (log) o dejarlos si son datos válidos y el modelo los tolera (árboles).

**Data leakage: el error de ajustar el scaler antes del split.** Es el error más frecuente y más difícil de detectar en la preparación de datos: cualquier transformación que "aprenda" algo de los datos (media y desvío de un scaler, la mediana para imputar, las categorías de un encoder) tiene que aprenderlo **solo del conjunto de entrenamiento**. Si se ajusta sobre todo el dataset antes de separar train/test, el conjunto de test "filtra" información hacia el entrenamiento (su media influyó en el escalado que ve el modelo) y las métricas de evaluación quedan artificialmente optimistas — el modelo funciona peor en producción de lo que el test hizo creer.

```python
# INCORRECTO — leakage: el scaler ve el test antes del split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # usa media/desvío de TODO el dataset
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

```python
# CORRECTO — el scaler solo aprende del train
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_tr)         # fit SOLO con train
X_tr_scaled = scaler.transform(X_tr)
X_te_scaled = scaler.transform(X_te)        # transform, no fit, sobre test
```

La forma robusta de garantizar esto en la práctica es un `Pipeline` (se retoma en el cap. 6), que encapsula el `fit` del scaler dentro de cada fold de validación cruzada.

```mermaid
flowchart TD
    A[Dataset completo] --> B[Split train / test]
    B --> C[Ajustar scaler / imputer SOLO con train]
    C --> D[Transformar train]
    C --> E[Transformar test con el mismo ajuste]
    D --> F[Entrenar modelo]
    F --> G[Evaluar sobre test transformado]
```

---

## Parte II — Cómo se evalúa un modelo

Tener un modelo entrenado no alcanza: la pregunta que importa es qué tan bien va a funcionar con datos que nunca vio. Esta parte cubre las herramientas para medir eso de forma honesta —train/test y validación cruzada—, el diagnóstico central de por qué un modelo falla —sobreajuste, subajuste y el compromiso sesgo-varianza— y las métricas concretas, separadas por tipo de problema, para poner un número a "qué tan bien". Es la base que hace falta antes de comparar clasificadores lineales, árboles o ensambles entre sí (partes siguientes): sin saber evaluar, no hay forma de decidir qué modelo es mejor.

### 6. Train/test y validación cruzada

**Por qué no alcanza con medir sobre el propio train.** Un modelo evaluado con los mismos datos con los que se entrenó siempre va a mostrar buen desempeño, aunque haya memorizado en vez de aprender — la métrica no distingue generalización de memorización. Por eso el paso mínimo es separar los datos en:

- **Train (entrenamiento):** con lo que el modelo ajusta sus parámetros.
- **Test (prueba):** apartado desde el principio, no se toca hasta la evaluación final; simula datos nuevos.

**Holdout vs k-fold.** El split simple train/test (**holdout**) tiene un problema: el resultado depende de qué observaciones cayeron al azar en cada lado del split, sobre todo con datasets chicos. La **validación cruzada k-fold** lo resuelve dividiendo el dataset en `k` partes (folds): se entrena `k` veces, cada vez dejando un fold distinto afuera como test y promediando el desempeño de las `k` corridas. El resultado es una estimación más estable y menos dependiente de un split particular.

```mermaid
flowchart TD
    A[Dataset completo] --> B[Dividir en k folds]
    B --> C[Iteración 1: fold 1 = test, resto = train]
    B --> D[Iteración 2: fold 2 = test, resto = train]
    B --> E[... hasta fold k]
    C --> F[Promediar métricas de las k iteraciones]
    D --> F
    E --> F
```

**`StratifiedKFold` y cuándo importa.** El k-fold simple parte los datos sin mirar la proporción de clases; con un target desbalanceado (ej. 90%/10%), un fold puede quedar con casi ninguna observación de la clase minoritaria, distorsionando la métrica de esa iteración. `StratifiedKFold` conserva en cada fold la misma proporción de clases que el dataset completo. Importa siempre en **clasificación** con clases desbalanceadas; en regresión no aplica (no hay clases que estratificar).

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
scores.mean(), scores.std()
```

**Por qué el escalado va dentro de un `Pipeline` durante el CV.** El mismo problema de leakage del cap. 5 reaparece en la validación cruzada: si el scaler se ajusta una sola vez sobre todo `X` antes de llamar a `cross_val_score`, cada fold de "test" ya fue visto por el scaler durante el ajuste, y las métricas de CV quedan optimistas. La solución es envolver preprocesamiento y modelo en un `Pipeline`: `cross_val_score` entonces re-ajusta el scaler (y cualquier otro paso) usando solo el train de cada fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")   # el scaler se re-ajusta en cada fold
```

### 7. Sobreajuste, subajuste y el compromiso sesgo-varianza

**Sobreajuste (overfitting):** el modelo memoriza particularidades del train —incluido el ruido— y generaliza mal. **Señal diagnóstica:** error de train muy bajo, error de validación/test notablemente más alto (la brecha entre ambos es la pista).

**Subajuste (underfitting):** el modelo es demasiado simple para capturar el patrón real. **Señal diagnóstica:** error de train ya alto, y el error de validación es similar (no hay brecha, pero ambos son malos).

| Diagnóstico | Error de train | Error de validación | Brecha |
|---|---|---|---|
| **Ajuste sano** | Bajo | Bajo | Chica |
| **Overfitting** | Bajo | Alto | Grande |
| **Underfitting** | Alto | Alto (similar al de train) | Chica |

**Descomposición sesgo-varianza, en palabras.** El error de generalización de un modelo se puede pensar como la suma de dos fuentes:

- **Sesgo (bias):** el error que viene de que el modelo es demasiado simple para el problema real — supuestos rígidos que no capturan la verdadera relación. Un modelo con sesgo alto **subajusta**: falla igual en train y en test, porque su límite no es la cantidad de datos sino su propia simpleza.
- **Varianza:** el error que viene de que el modelo es demasiado sensible a los datos particulares de entrenamiento — pequeños cambios en el train producen modelos muy distintos. Un modelo con varianza alta **sobreajusta**: le va muy bien en el train que memorizó y mal ante datos nuevos.

Los dos se mueven en tensión: bajar la complejidad reduce varianza pero sube sesgo, y viceversa. El objetivo no es eliminar ninguno de los dos sino encontrar el punto de **error de generalización mínimo**, que casi nunca coincide con el error de train mínimo.

```mermaid
flowchart LR
    A[Complejidad baja] --> B[Sesgo alto, varianza baja]
    B --> C[Underfitting]
    D[Complejidad alta] --> E[Sesgo bajo, varianza alta]
    E --> F[Overfitting]
    G[Complejidad óptima] --> H[Error de generalización mínimo]
```

**Curvas de aprendizaje.** Graficar el error de train y el de validación en función de la cantidad de datos de entrenamiento usados es la forma práctica de distinguir un caso del otro:

- Si ambas curvas convergen a un error **alto**: el problema es sesgo (underfitting) — más datos no ayudan por sí solos.
- Si hay una **brecha persistente** entre una curva de train baja y una de validación alta, incluso con mucho dato: el problema es varianza (overfitting).

**Qué palanca tocar en cada caso.**

| Diagnóstico | Palancas típicas |
|---|---|
| **Underfitting (sesgo alto)** | Modelo más complejo (más features, más profundidad, más capas), menos regularización, mejor feature engineering |
| **Overfitting (varianza alta)** | Más datos, **regularización** (cap. 15: Ridge/Lasso), simplificar el modelo, validación cruzada para elegir hiperparámetros, ensambles tipo bagging (cap. 14) |

> **Nota:** la regularización (cap. 15) es la herramienta principal contra el sobreajuste porque ataca directamente la varianza: penaliza la complejidad del modelo (magnitud de los coeficientes) sin necesitar más datos. En árboles, el equivalente es limitar hiperparámetros de crecimiento (cap. 13); en ensambles, el bagging (Random Forest) reduce varianza promediando modelos, y el boosting reduce sesgo entrenando secuencialmente (cap. 14).

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

Estas métricas se calculan siempre sobre el conjunto de **test** (o promediadas sobre los folds de CV, cap. 6), nunca sobre el train — de lo contrario se repite el mismo problema de optimismo artificial del cap. 7.

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_te, clf.predict(X_te)))
roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
```

### 9. Métricas de regresión

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

Al igual que en clasificación, estas métricas se calculan sobre test (o promediadas por CV), y la comparación entre el error de train y el de test es la forma directa de aplicar el diagnóstico del cap. 7 a un problema de regresión — la mecánica de fondo (regresión lineal, cap. 11; regularización, cap. 15) se desarrolla en las partes siguientes del manual.

## Parte III — Modelos lineales

Antes de entrar acá conviene tener frescos los conceptos de preparación de datos (cap. 5), validación cruzada (cap. 6) y el trade-off sesgo-varianza (cap. 7): un modelo lineal se entrena rápido, pero para juzgarlo bien hace falta separar train/test y mirar más allá del error de entrenamiento. Esta parte cubre la familia de modelos que separan clases o predicen valores con una combinación lineal de las features: primero los clasificadores lineales (regresión logística, SVM, y el propio perceptrón que se retoma en el cap. 26), después la regresión lineal para targets continuos, y por último cómo saber si un coeficiente estimado es real o ruido. Son los cimientos: todo lo que sigue en el manual —árboles, ensambles, regularización, redes— se entiende mejor en contraste con esta base lineal.

### 10. Clasificadores lineales

Los clasificadores lineales toman decisiones a partir de una **combinación lineal** de las features. Función de decisión:

```
f(x) = wᵀx + b
```

- `x`: vector de features · `w`: pesos · `b`: sesgo (bias).
- Clasificación binaria según el signo: `f(x) ≥ 0` → clase A; `f(x) < 0` → clase B.
- Un **hiperplano** es la frontera de decisión (recta en 2D, plano en 3D, etc.).

| Modelo | Idea clave |
|--------|-----------|
| **Perceptrón** | Clasificador lineal más simple; ajusta pesos según errores. Es el mismo tipo de frontera lineal que se retoma en detalle —estructura, entrenamiento manual y limitaciones— en el cap. 26. |
| **Regresión logística** | Pese al nombre, **clasifica**: modela `P(y=1\|X)` con la función logística (sigmoide), acotando la salida a `[0,1]`. |
| **SVM** | Busca el hiperplano que **maximiza el margen** entre clases (optimización cuadrática). |

**¿Por qué no usar regresión lineal para clasificar?** Daría valores fuera de `[0,1]`, imposibles de interpretar como probabilidad. La logística resuelve esto con la sigmoide:

```
σ(z) = 1 / (1 + e^(−z))   con z = wᵀx + b
```

**Regresión logística en detalle.** La salida `σ(z)` se interpreta como probabilidad de pertenecer a la clase positiva. Los coeficientes no se leen como en una regresión lineal (unidades de `y` por unidad de `x`), sino como **log-odds**: el modelo en realidad ajusta

```
log( P(y=1) / (1 − P(y=1)) ) = wᵀx + b
```

de modo que cada `wⱼ` representa el cambio en el **logaritmo de las odds** (la razón entre la probabilidad de la clase positiva y la de la negativa) por unidad de `xⱼ`, manteniendo el resto constante. Exponenciando `wⱼ` se obtiene el **odds ratio**: cuánto se multiplican las odds por cada unidad de la variable. Para decidir la clase final se aplica un **umbral de decisión** sobre `σ(z)` — por defecto `0.5`, pero se puede mover hacia arriba o abajo según el costo relativo de falsos positivos y falsos negativos (cap. 8), sin reentrenar el modelo.

**SVM (Support Vector Machines).** Busca, entre todos los hiperplanos que separan las clases, el que deja el **margen máximo** respecto a los puntos más cercanos de cada clase (los **vectores de soporte**). Cuando las clases no son linealmente separables en el espacio original, el **kernel trick** proyecta los datos a un espacio de mayor dimensión donde sí lo son, sin calcular esa proyección explícitamente — el kernel lineal (`linear`) mantiene la frontera lineal, y kernels como el radial (`rbf`) permiten fronteras curvas.

> **Nota:** perceptrón, regresión logística y SVM lineal comparten la misma forma de frontera de decisión —un hiperplano `wᵀx + b`—; lo que cambia es el criterio de ajuste: el perceptrón corrige por error, la logística maximiza verosimilitud, el SVM maximiza margen.

### 11. Regresión lineal

Predice una respuesta **cuantitativa** `Y` a partir de predictores `X`, asumiendo relación aproximadamente lineal.

**Simple:** `Y = β₀ + β₁·X + ε`
**Múltiple:** `Y = β₀ + β₁X₁ + … + βₚXₚ + ε`

- `β₀`: intercepto · `β₁…βₚ`: pendientes/coeficientes · `ε`: error.
- **Entrenar = estimar los coeficientes** (β̂) que mejor ajustan los datos.
- **Residuo:** `eᵢ = yᵢ − ŷᵢ`.
- Método de **mínimos cuadrados (OLS):** minimiza la suma de cuadrados de los residuos
  `RSS = Σ (yᵢ − ŷᵢ)²`.

Para evaluar qué tan bien ajusta el modelo se usan las métricas de regresión (cap. 9): MAE, MSE, RMSE y R². Ninguna de ellas dice, sin embargo, si un coeficiente en particular es estadísticamente distinto de cero — eso es tema del capítulo siguiente.

### 12. Inferencia sobre los coeficientes

**Test de significación individual** para `βⱼ`:

- **H₀:** `βⱼ = 0` (no hay relación entre `Xⱼ` e `Y`).
- **H₁:** `βⱼ ≠ 0` (hay relación).

Si `β₁ = 0`, el modelo se reduce a `Y = β₀ + ε` y `X` no aporta. Se evalúa con el **p-value**:

- **p-value chico** (< 0,05 típicamente) → se rechaza H₀: hay evidencia de relación.
- También se usan **intervalos de confianza** para los coeficientes.

Esta inferencia es la base para decidir, más adelante, qué variables conservar en un modelo (selección de variables, cap. 16) y para comparar modelos completos entre sí con criterios como AIC/BIC (cap. 17): un coeficiente no significativo es candidato a eliminarse.

---

## Parte IV — Modelos no lineales

Los modelos lineales de la Parte III trazan fronteras rectas (o hiperplanos); cuando la relación entre features y target no es lineal, hace falta otra familia de modelos. Esta parte cubre los árboles de decisión —que particionan el espacio con preguntas sucesivas— y los ensambles que se construyen combinando muchos árboles (u otros modelos débiles) para ganar precisión y estabilidad. Conviene tener frescos el sobreajuste y el trade-off sesgo-varianza (cap. 7) y las métricas de clasificación (cap. 8), porque son la vara con la que se juzga si un árbol o un ensamble realmente mejora las cosas.

### 13. Árboles de decisión

Modelo predictivo con estructura jerárquica: **nodo raíz → nodos internos (preguntas) → ramas (respuestas) → hojas (predicción)**. Sirve para clasificación y regresión (**CART**).

```mermaid
flowchart TD
    A[Edad menor a 30] -->|Si| B[Ingreso menor a 50000]
    A -->|No| C[Hoja: Aprobado]
    B -->|Si| D[Hoja: Rechazado]
    B -->|No| E[Hoja: Aprobado]
```

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

El sobreajuste y la inestabilidad se mitigan con **ensambles** (Random Forest, Gradient Boosting / XGBoost), tema del capítulo siguiente.

### 14. Ensambles y XGBoost

**Qué es un ensamble y por qué funciona.** Un ensamble combina las predicciones de **varios modelos débiles** (típicamente árboles poco profundos) para producir un modelo fuerte. Funciona porque los errores de modelos distintos, entrenados de forma distinta, tienden a no estar perfectamente correlacionados: al promediar (o combinar) sus predicciones, los errores individuales se cancelan parcialmente y queda una señal más estable. Es la misma lógica que "el promedio de muchas estimaciones ruidosas es más preciso que una sola estimación": reduce la varianza del conjunto, mejora el sesgo, o ambas cosas, según cómo se combinen los modelos.

**Bagging vs boosting — la diferencia conceptual.**

- **Bagging** (*bootstrap aggregating*, ej. Random Forest): entrena **muchos árboles en paralelo**, cada uno sobre una muestra bootstrap distinta (con reemplazo) del dataset original, y promedia sus predicciones (o vota, en clasificación). Como los árboles se entrenan de forma independiente y se promedian, lo que se gana es **reducción de varianza**: un árbol individual sobreajusta fácil, pero el promedio de muchos árboles distintos generaliza mejor.
- **Boosting** (ej. Gradient Boosting, XGBoost): entrena los árboles **secuencialmente**, y cada árbol nuevo se enfoca en corregir los **errores (residuos)** que dejó el árbol anterior. Como cada paso ataca específicamente lo que el modelo acumulado todavía no explica, lo que se gana es **reducción de sesgo**: modelos individuales simples (poco profundos) se van sumando hasta capturar relaciones complejas que ninguno por separado podría.

```mermaid
flowchart TB
    subgraph Bagging
    D1[Muestra bootstrap 1] --> M1[Arbol 1]
    D2[Muestra bootstrap 2] --> M2[Arbol 2]
    D3[Muestra bootstrap 3] --> M3[Arbol 3]
    M1 --> P1[Promedio o voto]
    M2 --> P1
    M3 --> P1
    end
    subgraph Boosting
    T1[Arbol 1] --> R1[Errores del arbol 1]
    R1 --> T2[Arbol 2 corrige R1]
    T2 --> R2[Errores del arbol 2]
    R2 --> T3[Arbol 3 corrige R2]
    T3 --> P2[Suma ponderada]
    end
```

**Random Forest.** Ensamble por bagging de árboles de decisión, con un ingrediente extra: en cada split, en vez de considerar todas las features, elige la mejor división entre un **subconjunto aleatorio** de ellas (`max_features`). Esto decorrelaciona más los árboles entre sí (si una feature es muy fuerte, sin este truco todos los árboles la usarían primero y se parecerían demasiado), lo que mejora aún más la reducción de varianza. Da además `feature_importances_` como subproducto útil para EDA.

**Gradient Boosting y XGBoost.** El *gradient boosting* generaliza la idea de boosting: cada árbol nuevo se ajusta al **gradiente** de la función de pérdida respecto a las predicciones actuales (en el caso más simple, ajusta directamente los residuos). **XGBoost** (*Extreme Gradient Boosting*) es una implementación optimizada de gradient boosting: agrega regularización incorporada (evita que los árboles individuales crezcan demasiado y sobreajusten), maneja datos faltantes de forma nativa, y está optimizado en velocidad y uso de memoria. Es el estándar de facto en problemas tabulares (competencias, producción).

**Hiperparámetros principales de XGBoost / Gradient Boosting:**

| Hiperparámetro | Qué controla |
|---|---|
| `n_estimators` | Cantidad de árboles secuenciales. Más árboles → más capacidad, pero más riesgo de sobreajuste y más tiempo de entrenamiento. |
| `learning_rate` | Cuánto pesa la corrección de cada árbol nuevo sobre el total. Valores chicos (ej. 0.01-0.1) generalizan mejor pero necesitan más `n_estimators` para compensar. |
| `max_depth` | Profundidad máxima de cada árbol individual. En boosting suele mantenerse bajo (3-8): los árboles son "débiles" a propósito; la fuerza viene de sumarlos, no de que cada uno sea complejo. |
| `subsample` | Fracción de las filas usada para entrenar cada árbol (muestreo sin reemplazo). Menor a 1 agrega aleatoriedad tipo bagging dentro del boosting, reduce varianza y overfitting. |

> **Nota:** `learning_rate` y `n_estimators` interactúan: bajar el `learning_rate` casi siempre exige subir `n_estimators` para llegar al mismo nivel de ajuste, a cambio de un modelo final más robusto.

**Cuándo un ensamble no se justifica.** Un ensamble agrega costo de entrenamiento, tiempo de inferencia y pérdida casi total de interpretabilidad (cap. 13 lista la interpretabilidad como ventaja del árbol simple; se pierde acá). No conviene cuando: el dataset es muy chico (no hay margen para que el ensamble generalice mejor que un modelo simple bien regularizado); se necesita explicar cada predicción a un humano (auditoría, decisiones regulatorias) y no alcanza con `feature_importances_` o SHAP; el problema ya es linealmente separable y un modelo lineal (cap. 10-11) alcanza el mismo desempeño con muchísima menos complejidad; o los recursos de cómputo/latencia en producción son muy limitados (un árbol único o un modelo lineal responden en microsegundos; un ensamble grande, no siempre).

---

## Parte V — Complejidad del modelo y selección

Con modelos lineales y no lineales ya cubiertos, queda una pregunta transversal a ambas familias: ¿cuántas variables usar, y cuánto restringir los coeficientes para que el modelo generalice? Esta parte conecta directamente con el sobreajuste (cap. 7): regularización y selección de variables son las dos herramientas principales para controlar la complejidad de un modelo lineal, y los criterios AIC/BIC dan una forma numérica de comparar modelos completos entre sí. También son la base conceptual de temas posteriores del manual: k-means y reducción de dimensionalidad (cap. 18 y 23) enfrentan el mismo problema de "cuántas dimensiones conservar" desde el lado no supervisado, y la regularización en redes neuronales (cap. 36) reutiliza exactamente las ideas de norma L1/L2 que se explican acá.

### 15. Regularización: Ridge, Lasso y Elastic Net

Añade una **penalización** al costo de entrenamiento: `Costo = RSS + α·penalización`. El hiperparámetro **α (o λ)** regula la fuerza de la penalización y se elige por **cross-validation** (cap. 6). Es imprescindible **estandarizar** las variables antes de regularizar, porque la penalización actúa sobre la magnitud de los coeficientes, y esa magnitud depende de la escala de cada variable.

**La idea central — norma L1 vs L2.** La penalización se calcula como una norma sobre el vector de coeficientes `β`, y el tipo de norma determina qué le hace al modelo:

- **Norma L2** (`Σβ²`, usada por **Ridge**): penaliza el cuadrado de cada coeficiente. Como el cuadrado crece muy rápido cuando el coeficiente es grande y muy poco cuando ya es chico, el efecto es **achicar todos los coeficientes de forma proporcional**, acercándolos a cero sin llegar nunca exactamente a cero. Ridge no elimina variables: las atenúa a todas.
- **Norma L1** (`Σ|β|`, usada por **Lasso**): penaliza el valor absoluto de cada coeficiente. A diferencia del cuadrado, la penalización L1 crece de forma lineal y tiene una "esquina" en cero que hace que, al optimizar, algunos coeficientes terminen exactamente en **cero** — no solo chicos, sino eliminados del modelo. Por eso Lasso hace **selección automática de variables** como efecto colateral de regularizar.
- **Elastic Net** combina ambas penalizaciones (`α₁·L1 + α₂·L2`, o parametrizado como una mezcla `l1_ratio`): captura algo de la selección de Lasso y algo de la estabilidad de Ridge, útil cuando hay variables correlacionadas entre sí (Lasso tiende a elegir arbitrariamente una de un grupo correlacionado; Elastic Net reparte entre ellas).

| Método | Norma | ¿Anula coeficientes? | Cuándo usar |
|--------|:-----:|:--------------------:|-------------|
| **Ridge** | L2 (`Σβ²`) | **No** (los achica) | Colinealidad; conservar todas las variables |
| **Lasso** | L1 (`Σ\|β\|`) | **Sí** (a cero exacto) | **Selección automática**; modelos dispersos |
| **Elastic Net** | L1 + L2 | Sí (parcial) | Combina ambas; **2 hiperparámetros** (λ, α) |

```mermaid
flowchart LR
    A[Coeficientes sin regularizar] --> B{Tipo de norma}
    B -->|L2 Ridge| C[Todos los coeficientes se achican]
    B -->|L1 Lasso| D[Algunos coeficientes llegan a cero]
    C --> E[Modelo denso mas estable]
    D --> F[Modelo disperso con seleccion de variables]
```

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_scaled, y)   # elige alpha por CV
lasso.coef_    # los que quedan en 0 fueron descartados
```

> **Esta es la explicación canónica de regularización en el manual.** El cap. 36 (regularización en redes neuronales) parte de esta misma idea de norma L1/L2 sobre los pesos y agrega solo lo específico de las redes (dropout, early stopping); la lógica de fondo —L2 achica, L1 elimina— es la misma acá y ahí.

### 16. Selección de variables

Elegir un **subconjunto** de las features originales. Mejora rendimiento, interpretabilidad, reduce overfitting y acelera el entrenamiento. A diferencia de la regularización (cap. 15), que penaliza coeficientes dentro del entrenamiento, estos métodos deciden qué variables entran al modelo antes o durante el ajuste, con distintos criterios.

**Filtros (filter)** — medida estadística, sin entrenar modelo (rápidos, antes de modelar):

| Método | Problema | Idea |
|--------|----------|------|
| **Correlación (Pearson)** | Regresión | features con coef. cercano a ±1 con el target |
| **Chi-cuadrado (χ²)** | Clasificación, categóricas | mayor dependencia feature-clase = más relevante |
| **ANOVA (F-test)** | Clasificación, numéricas | mayor F = mejor separa clases |
| **Coeficientes lineales** | Regresión/logística | mayor \|coef\| = más influyente |

**Wrapper — RFE (Eliminación Recursiva):** entrena un modelo, elimina la feature menos importante, reentrena y repite. Preciso pero **caro** (entrena en cada iteración).

**Embedded:** la selección ocurre dentro del entrenamiento (árboles/Random Forest vía `feature_importances_`, cap. 13-14; o Lasso, cap. 15, que anula coeficientes). Mantiene las features originales, a diferencia de técnicas que crean nuevas variables combinando las originales (reducción de dimensionalidad, cap. 23).

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)   # filtro ANOVA
```

| Criterio | Filtros | RFE |
|----------|---------|-----|
| Depende de un modelo | No | **Sí** |
| Costo | Bajo | Alto (iterativo) |
| Cuándo | Screening rápido | Optimizar modelo final |

```mermaid
flowchart TD
    A[Todas las features] --> B{Metodo de seleccion}
    B -->|Filtro| C[Correlacion Chi2 ANOVA]
    B -->|Wrapper| D[RFE entrena y elimina]
    B -->|Embedded| E[Lasso o feature importances]
    C --> F[Subconjunto final]
    D --> F
    E --> F
```

### 17. Criterios de selección de modelos (AIC y BIC)

Comparan modelos equilibrando **ajuste vs complejidad** (penalizan el número de parámetros). **Menor = mejor.** Mientras que la regularización (cap. 15) y la selección de variables (cap. 16) actúan **dentro** del proceso de ajuste de un modelo, AIC y BIC sirven para comparar **modelos ya ajustados entre sí** —por ejemplo, para decidir entre dos regresiones con distinto número de predictores, sin necesidad de un set de test separado.

| | AIC | BIC |
|--|-----|-----|
| **Fórmula** | `2k − 2·ln(L)` | `ln(n)·k − 2·ln(L)` |
| **Penalización** | `2` por parámetro (fija) | `ln(n)` (crece con la muestra) |
| **Tendencia** | Admite más complejidad | Más conservador (modelos simples) |

`k` = nº parámetros, `n` = nº observaciones, `L` = máxima verosimilitud. Usos: comparar modelos de regresión, ARIMA, o el nº de componentes en un **GMM**.

> **Nota:** con `n` grande, la penalización de BIC (`ln(n)`) supera a la de AIC (constante `2`), por lo que BIC tiende a preferir modelos más simples que AIC a medida que crece el dataset.

## Parte VI — Aprendizaje no supervisado

Hasta acá todo el manual asumió que había un `y` para entrenar y evaluar (cap. 8 y 9). Esta parte cambia esa premisa: no hay etiquetas, y el objetivo es encontrar estructura en los datos por su cuenta. Para aprovecharla hace falta tener asimilada la preparación de datos (cap. 5) — en particular el escalado, porque casi todo lo que sigue trabaja con distancias o con varianza — y conviene tener presente el problema de sobreajuste y sesgo-varianza (cap. 7), porque reaparece bajo otra forma cuando no hay `y` para chequear contra el mundo real.

### 18. Clustering con K-means

**Apuntes.** Particiona los datos en **k** clusters minimizando la **inercia** (suma de distancias² de cada punto a su centroide). Iterativo: inicializa k centroides → asigna cada punto al más cercano → recalcula centroides como la media → repite hasta converger.

**Referencia técnica.**

| | |
|--|--|
| **Cuándo usar** | Sabés (o estimás) k; grupos aproximadamente **esféricos** y de tamaño similar; dataset grande (escala bien). |
| **Cuándo NO** | Formas arbitrarias, densidades/tamaños muy distintos, outliers (los arrastra). |
| **Hiperparámetros** | `n_clusters` (**el** clave), `init` (`k-means++`), `n_init`, `max_iter`. |
| **Escala** | Sensible → estandarizar antes (cap. 5). |
| **Evaluación** | Codo (inercia), silueta, Davies-Bouldin (cap. 21). |

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
km.inertia_   # WCSS para el método del codo
```

### 19. Clustering jerárquico

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

### 20. DBSCAN: clustering por densidad

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

Un árbol de decisión rápido para elegir entre los tres, según la forma de los datos y si conocés k de antemano:

```mermaid
flowchart TD
    A[Necesito agrupar datos] --> B{Conozco k?}
    B -->|Si, grupos esfericos| C[K-means]
    B -->|No| D{Quiero ver jerarquia?}
    D -->|Si, dataset chico| E[Jerarquico]
    D -->|No| F{Formas arbitrarias con ruido?}
    F -->|Si| G[DBSCAN]
    F -->|No| C
```

### 21. Evaluación de clusters

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

### 22. La maldición de la dimensión

Problemas en **alta dimensión**: los datos se **dispersan**, las distancias se vuelven **uniformes** y pierden significado (degradan K-means/K-NN, cap. 10); el costo crece exponencialmente; sube el riesgo de **overfitting** (cap. 7); la visualización se vuelve imposible.

**Mitigaciones:** reducción de dimensionalidad (cap. 23), selección de variables, regularización (cap. 15). Es el puente entre selección de variables y reducción de dimensionalidad.

### 23. Reducción de dimensionalidad

**Crean** nuevas variables que preservan información (distinto de seleccionar un subconjunto de las originales).

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

| Necesito… | Técnica | Cap. |
|-----------|---------|:-:|
| Agrupar, sé k, grupos esféricos | K-means | 18 |
| Agrupar y ver estructura jerárquica | Jerárquico | 19 |
| Agrupar formas raras + detectar outliers | DBSCAN | 20 |
| Elegir nº óptimo de clusters | Codo + Silueta | 21 |
| Reducir dimensión conservando varianza | PCA | 23 |
| Reducir dimensión separando clases | LDA | 23 |
| Visualizar alta dimensión en 2D (dataset chico) | t-SNE | 23 |
| Visualizar alta dimensión en 2D (dataset grande) | UMAP | 23 |

## Parte VII — Redes neuronales

Esta parte deja atrás los modelos clásicos (regresión, árboles, clustering) para entrar en redes neuronales, que necesitan del resto del manual como base: los clasificadores lineales (cap. 10) para entender qué es exactamente el perceptrón y por qué es un caso particular de ellos, la regularización Ridge/Lasso (cap. 15) porque reaparece igual en las redes, y el sobreajuste (cap. 7) porque es el riesgo central de un MLP. El recorrido va de lo biológico y lo histórico hasta el mecanismo interno de entrenamiento — backpropagation y descenso de gradiente — pasando por el perceptrón simple, sus límites, y el perceptrón multicapa que los supera.

> **Nota:** curso de redes neuronales documentado hasta la Clase 11 del programa: perceptrón, su entrenamiento manual con la compuerta AND, limitaciones, implementación con scikit-learn (`Perceptron`) y perceptrón multicapa (`MLPClassifier`, `MLPRegressor`). El material del curso sigue en curso de publicación.

### 24. Fundamentos biológicos

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

### 25. Historia de las redes neuronales

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

### 26. El perceptrón: estructura y fórmulas

**Apuntes.** El **perceptrón** (Rosenblatt, 1957/58) es un algoritmo de **aprendizaje supervisado** para **clasificación binaria**, el modelo más simple de red neuronal. Es un **clasificador lineal** (cap. 10): solo puede aprender correctamente cuando las clases son **linealmente separables** (se pueden dividir con una única recta/hiperplano).

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

### 27. Entrenamiento del perceptrón: la compuerta AND

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

### 28. Limitaciones del perceptrón

**Apuntes.**

- **Separabilidad lineal:** el perceptrón simple **solo** resuelve problemas **linealmente separables**. El caso clásico que no puede resolver es el **XOR**, cuyas clases no se pueden separar con una única recta/hiperplano.
- **Convergencia no garantizada:** si los datos no son linealmente separables, el algoritmo puede no converger nunca en un número finito de pasos.
- **Capacidad limitada:** poca capacidad para capturar relaciones y patrones complejos (al ser un modelo lineal de una sola capa).
- **Ajuste de hiperparámetros:** aunque tiene menos hiperparámetros que modelos más complejos, calibrar la tasa de aprendizaje sigue siendo un desafío (muy alta → inestabilidad/oscilación; muy baja → convergencia lenta).

Esta limitación (XOR) es históricamente la que originó el "invierno de la IA" (cap. 25): se resuelve con perceptrones **multicapa** (MLP) y **backpropagation**, contenido de los capítulos siguientes.

### 29. Implementación con scikit-learn

**Apuntes.**

- **Scikit-Learn** es una de las librerías más usadas del ecosistema Python para **aprendizaje automático**.
- Ofrece una **amplia gama de algoritmos y utilidades** que cubren **todo el flujo de trabajo**: desde la preparación de los datos hasta la evaluación del modelo.
- **Se integra** con el resto del ecosistema Python, incluidas las librerías de aprendizaje profundo como **TensorFlow** y **PyTorch**.

Aplicado a este módulo: el entrenamiento manual de la compuerta AND (cap. 27) queda encapsulado en `sklearn.linear_model.Perceptron`, que implementa la misma regla delta detrás de la API uniforme `fit` / `predict` (ver ejemplo en la referencia técnica de más abajo). La implementación manual sirve para entender la mecánica; scikit-learn, para trabajar en la práctica con validación, métricas y preprocesamiento integrados. Para redes profundas (varias capas, backpropagation) se pasa a TensorFlow o PyTorch.

### 30. El perceptrón multicapa (MLP)

**Apuntes.**

Un **perceptrón multicapa (MLP)** es una red neuronal artificial que aborda problemas complejos de **clasificación** y **regresión**. A diferencia del perceptrón simple, que solo resuelve problemas **linealmente separables** (cap. 28), el MLP maneja **relaciones no lineales** entre las características de entrada y la salida.

Los cuatro puntos de la slide:

- El entrenamiento se realiza mediante el algoritmo de **retropropagación** (backpropagation, ver cap. 37).
- Su capacidad para aprender relaciones complejas viene de la **estructura de capas** y de las **funciones de activación no lineales**.
- Las redes multicapa son **sensibles a la calidad y la naturaleza de los datos** de entrada.
- Necesitan **muchos datos** para entrenar eficazmente y tienen **riesgo de sobreajuste** (cap. 7).

**Arquitectura.**

| Capa | Rol |
|---|---|
| **Entrada** | una neurona por característica; no calcula, solo recibe |
| **Ocultas** (1 o más) | dan la capacidad de representar relaciones no lineales; cada neurona hace `z = Σ wᵢ·xᵢ + b` y le aplica una activación no lineal |
| **Salida** | una neurona en regresión; una por clase en clasificación |

```mermaid
flowchart LR
    subgraph Entrada
        x1((x1))
        x2((x2))
        x3((x3))
    end
    subgraph Oculta
        h1((h1))
        h2((h2))
        h3((h3))
    end
    subgraph Salida
        y1((y))
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    x3 --> h1
    x3 --> h2
    x3 --> h3
    h1 --> y1
    h2 --> y1
    h3 --> y1
```

Sobre esa arquitectura corren dos procesos:

- **Forward propagation:** los datos de entrada se transforman a través de las capas hasta producir la **salida final**.
- **Backpropagation:** ajuste de **pesos y sesgos** para **minimizar la función de pérdida**. Es la generalización a varias capas de la regla delta del entrenamiento manual (cap. 27). Se desarrolla en detalle en el cap. 37.

**Por qué la activación oculta debe ser no lineal.** Es el punto que justifica toda la arquitectura: si las capas ocultas solo hicieran la suma ponderada, la composición de varias capas lineales **seguiría siendo lineal** y la red colapsaría al equivalente de un perceptrón simple, con la misma limitación. La no linealidad (ReLU, tanh, logística — ver cap. 32) es lo que hace que apilar capas agregue capacidad real, y con eso resolver el **XOR**.

**Qué se paga respecto del perceptrón simple.**

| | Perceptrón simple | MLP |
|---|---|---|
| Problemas | solo linealmente separables | también no lineales |
| Interpretabilidad | alta: 2 pesos y un sesgo legibles | baja: miles de pesos sin lectura directa |
| Datos necesarios | pocos | muchos |
| Sobreajuste | bajo (modelo rígido) | alto: requiere regularización (`alpha`, cap. 36) y validación |
| Escalado de entradas | tolerable | **necesario** |
| Costo de entrenamiento | trivial | significativo |

> **Nota (verificado en notebook):** sobre Iris con 2 atributos y las 3 clases, el perceptrón simple llega a **76,7%** de exactitud y el MLP a **93,3%** — la diferencia es exactamente el solapamiento entre *versicolor* y *virginica*, que ninguna recta separa. En el mismo experimento, una red de **20** neuronas iguala a una de **250** usando dos órdenes de magnitud menos parámetros, mientras que una de **5** se queda corta (80%): el tamaño de la red es un hiperparámetro **a buscar**, no a maximizar.

### 31. Grafos y capa densa

**Apuntes.**

Un **grafo** describe cómo están conectadas las unidades: **nodos** (neuronas) y **aristas** (conexiones, cada una con su peso). En una **capa densa** o **totalmente conectada**, **cada neurona está conectada a todas las neuronas de la capa anterior**.

Consecuencia práctica: entre una capa de `n` neuronas y una densa de `m` hay `n × m` pesos más `m` sesgos. Por eso los parámetros crecen como un **producto**, no como una suma, al agrandar la red.

El diagrama del curso rotula cada neurona oculta como **`Σf`**, y el rótulo es literal: **suma ponderada** (`Σ`) seguida de la **función de activación** (`f`). Es decir, cada nodo es un perceptrón (cap. 26); la red es muchos de ellos conectados. En `scikit-learn`, `hidden_layer_sizes=(100, 150)` describe exactamente ese grafo, y los pesos quedan en `mlp.coefs_` y los sesgos en `mlp.intercepts_`.

### 32. Funciones de activación

**Apuntes.**

La **función de activación** determina la salida de una neurona. Sin ella, la red es una **combinación lineal de las entradas** y apilar capas no agrega nada (cap. 30). Sus cuatro roles según la slide: introducen **no linealidad**, deciden si la neurona **se activa** (pasa información a la siguiente capa), algunas **normalizan la salida a un rango**, y **afectan cómo se actualizan los pesos** —porque la retropropagación usa su derivada—.

| Función | Fórmula | Rango | Nota |
|---|---|---|---|
| **Sigmoide (logística)** | `1 / (1 + e^(−x))` | (0, 1) | salida legible como probabilidad; **satura** en los extremos |
| **Tanh** | `(e^z − e^(−z)) / (e^z + e^(−z))` | (−1, 1) | igual forma pero **centrada en 0**; converge más rápido; también satura |
| **ReLU** | `0 si x < 0; x si x ≥ 0` | [0, ∞) | derivada 1 en el lado positivo: **no satura**; barata; default en capas ocultas |

> **Errata del material:** la slide escribe la sigmoide como `1/(1 − e^(−x))`. Es un error tipográfico: va **más**. Con el signo menos la función se indefine en `x = 0` y no coincide con la curva dibujada en la misma slide.

**Saturación y gradiente desvaneciente.** En los extremos, sigmoide y tanh son casi planas: su derivada tiende a 0. Como la retropropagación **multiplica** derivadas capa por capa (cap. 37), en redes profundas el gradiente se apaga y las primeras capas dejan de aprender. ReLU no tiene ese problema por el lado positivo; a cambio, una neurona cuya entrada queda siempre negativa tiene gradiente 0 y muere (*dying ReLU*).

> **Nota (verificado en notebook):** la derivada en `x = 6` vale **0,0025** para la sigmoide y **0,00002** para tanh, contra **1** para ReLU: la saturación es medible, no una figura retórica. Y sobre el XOR, un `MLPClassifier` con `activation='identity'` se queda en **50%** de exactitud —no lo resuelve—, mientras que con `tanh` o `relu` llega a **100%**. La no linealidad es lo único que cambia entre esos tres casos.

**Qué usar:** ReLU en capas ocultas por defecto; **sigmoide** en la salida binaria; **softmax** en la salida multiclase; **ninguna** (identidad) en la salida de regresión. En `scikit-learn`, `activation` aplica **solo a las capas ocultas** —la de salida la elige el estimador según el problema— y acepta `'relu'` (default), `'tanh'`, `'logistic'`, `'identity'`.

### 33. Diseño de la arquitectura de la red

**Apuntes.**

Elegir el **número de capas** y de **neuronas** afecta directamente la **capacidad de aprendizaje**, el **poder de generalización** y el **rendimiento computacional**.

| Decisión | Si se pasa | Si se queda corto |
|---|---|---|
| **Capas** | más complejidad y más tiempo de entrenamiento | no aprende representaciones complejas |
| **Neuronas** | **sobreajuste** (cap. 7): memoriza el train | **capacidad insuficiente** para la complejidad de los datos |

No hay fórmula: la arquitectura es un **hiperparámetro** y se busca comparando en validación (cap. 6), nunca en train.

> **Nota (verificado en notebook, Iris con 2 atributos, `relu`):** `(2,)` da 46,7% de train y 56,7% de test con 15 parámetros —**underfitting** de manual: no ajusta ni el entrenamiento—; `(100,)` llega a 100% de test con 603 parámetros; y `(100, 100)`, con **10.703** parámetros (~17 veces más), **no mejora nada**: baja a 96,7%. La lectura honesta es "capacidad de sobra sin beneficio" más que sobreajuste probado, porque con **30 muestras de test** un acierto vale 3,3 puntos y las diferencias chicas no son concluyentes.

> **Nota (verificado en notebook):** escalar con `StandardScaler` **bajó la pérdida final en las tres activaciones** (tanh 0,243 → 0,161; relu 0,197 → 0,177), pero el efecto sobre la exactitud fue **mixto**: `logistic` anduvo mejor *sin* escalar (100% contra 90%). Ninguna de las seis corridas convergió dentro de `max_iter=300`. El escalado es buena práctica por lo que le hace al entrenamiento, no porque garantice mejor exactitud en un dataset chico. Método práctico: arrancar simple (una capa oculta), agrandar solo si el error de *entrenamiento* sigue alto, y si el error de train es bajo pero el de test alto, achicar o regularizar (`alpha`, cap. 36) antes que agregar capas. Verificar lo que realmente quedó entrenado con `mlp.n_layers_`, `mlp.coefs_` y `mlp.loss_curve_`.

### 34. Funciones de pérdida

**Apuntes.**

La **función de pérdida** (o **función de costo**) cuantifica la **discrepancia entre lo que el modelo predijo y los valores reales**, produciendo **un único valor**. Ese valor es la **señal para ajustar pesos y sesgos**: es lo que la retropropagación deriva (cap. 37). En `scikit-learn` es la curva `mlp.loss_curve_`. La elección **depende del tipo de problema**.

**Regresión.**

```
MAE = (1/n) · Σ |yᵢ − ŷᵢ|          MSE = (1/n) · Σ (yᵢ − ŷᵢ)²
```

El MSE eleva al cuadrado: castiga mucho más los errores grandes y es **sensible a outliers**; el MAE es más **robusto**. A cambio, el MSE es derivable en todo su dominio, lo que lo hace más cómodo para el descenso de gradiente. `MLPRegressor` minimiza MSE. (Ver métricas de regresión, cap. 9.)

**Clasificación.**

```
BCE:  L = −(1/N) · Σᵢ ( yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) )
CCE:  L = −(1/N) · Σⱼ Σᵢ  yⱼᵢ·log(ŷⱼᵢ)
```

La **entropía binaria cruzada (BCE)** se usa en clasificación **binaria**; la **entropía cruzada categórica (CCE)**, en **multiclase**. Ambas miden la diferencia entre la distribución verdadera y la predicha. Con `y` en formato *one-hot*, en la CCE solo sobrevive el término de la clase correcta: la pérdida es `−log` de la probabilidad que el modelo le asignó a esa clase. `MLPClassifier` **no expone el parámetro**: usa log-loss siempre, que es BCE en el caso binario y CCE en el multiclase. (Ver métricas de clasificación, cap. 8.)

**Pérdida ≠ métrica de evaluación.** La pérdida es lo que se **minimiza** durante el entrenamiento y debe ser derivable; la métrica (exactitud, R², RMSE) es lo que se **reporta**. A veces coinciden (MSE) y a veces no: nadie entrena minimizando exactitud, porque no es derivable.

### 35. Optimización y descenso de gradiente

**Apuntes.**

**Optimizar** es ajustar pesos y sesgos para **minimizar la función de pérdida** (cap. 34). El método es el **descenso de gradiente**: se calcula el **gradiente** —la derivada de la función de error respecto de *todos* los parámetros de la red, calculada con backpropagation (cap. 37)— y se avanza en sentido contrario.

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

El flujo completo de una época de entrenamiento, de los datos a la actualización de pesos:

```mermaid
flowchart LR
    A[Datos de entrada] --> B[Forward propagation]
    B --> C[Prediccion]
    C --> D[Funcion de perdida]
    D --> E[Backpropagation]
    E --> F[Gradiente por parametro]
    F --> G[Actualizar pesos y sesgos]
    G --> A
```

**Dónde encaja:** la **regla delta** del perceptrón (cap. 27) es este mismo mecanismo en su versión mínima; **backpropagation** (cap. 37) es el algoritmo que calcula ese gradiente para todas las capas aplicando la regla de la cadena hacia atrás; y la **saturación** de las activaciones (cap. 32) es lo que lo rompe, porque multiplica el gradiente por números casi nulos.

### 36. Regularización en redes neuronales

**Apuntes.**

La explicación canónica de regularización —qué es la norma L1 y L2, y qué le hace cada una a los coeficientes— ya se desarrolló para modelos lineales en el cap. 15 (Ridge y Lasso). En redes neuronales es la **misma matemática** aplicada a los pesos de las capas, y solo cambia lo que aporta cada método frente a las particularidades de un MLP:

| Técnica | Penalización | Efecto sobre los pesos |
|---|---|---|
| **L1** | `L(X, w) + λ · Σ \|wᵢ\|` — suma de **valores absolutos** | lleva pesos **a cero**: selecciona variables, red rala |
| **L2** | `L(X, w) + λ · Σ wᵢ²` — suma de **cuadrados** | los **encoge** hacia cero sin anularlos |
| **Dropout** | — | **apaga neuronas al azar** durante el entrenamiento |

**Dropout** es lo propio de las redes, sin equivalente en regresión lineal: no toca la función de pérdida sino la arquitectura durante el entrenamiento, y al apagar neuronas al azar impide que la red dependa de una neurona en particular.

| Ventajas | Desventajas |
|---|---|
| previene el sobreajuste | más costo computacional |
| controla la complejidad | suma hiperparámetros que hay que elegir |
| puede acelerar la convergencia | posible pérdida parcial de información |

**En `scikit-learn` solo hay L2**, vía `alpha` (la `λ` de la fórmula del cap. 15), con default `0.0001`. **No hay L1 ni dropout** para redes: eso requiere Keras o PyTorch. Sí está disponible la **parada temprana** (`early_stopping=True`), que es regularización de hecho y es propia del entrenamiento iterativo de una red (no tiene equivalente directo en Ridge/Lasso, que se resuelven en una sola pasada): cortar cuando la validación deja de mejorar evita seguir ajustando ruido.

**Cómo se busca `alpha`:** con validación cruzada (cap. 6), graficando exactitud de **train y de validación** contra `alpha`. La señal de sobreajuste es la **brecha** entre las dos curvas; el buen `alpha` la cierra sin hundir las dos.

> **Nota (verificado en el recurso de clase):** un barrido de `alpha` sobre un problema demasiado fácil **no muestra nada**. En `OD_RN1_ESP_M03_S10`, cinco valores de `alpha` que cubren cinco órdenes de magnitud (de 0,00001 a 1,0) dan **todos la misma exactitud, 0,97**, y los tres solvers también. Con Iris de 2 atributos y 30 muestras de test, la exactitud solo puede valer 29/30 o 30/30: no hay resolución para distinguir configuraciones. Para ver el efecto de la regularización hace falta un problema que efectivamente sobreajuste.

### 37. Backpropagation



**La idea.** Entrenar una red es encontrar los pesos que minimizan la función de pérdida (cap. 34). Para eso hace falta el gradiente de la pérdida respecto de *cada* peso de *cada* capa (cap. 35). El problema es que la salida de la red es una composición de funciones —capa tras capa— y un peso de una capa temprana afecta la pérdida solo indirectamente, a través de todas las capas que vienen después. **Backpropagation** es el algoritmo que calcula ese gradiente completo de forma eficiente, propagando el error desde la salida hacia atrás, capa por capa, hasta la entrada.

**La regla de la cadena.** Es la herramienta matemática que lo hace posible. Si la pérdida `L` depende de la salida `y`, que depende de `z` (la suma ponderada), que depende de un peso `w`, entonces:

```
∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
```

Cada capa aporta un factor a esa cadena de derivadas. Para un peso de una capa intermedia, la cadena simplemente se hace más larga: incluye un factor por cada capa que hay entre ese peso y la salida.

**Dos pasadas por la red.**

| Paso | Dirección | Qué hace | Qué guarda |
|---|---|---|---|
| **Forward** | entrada → salida | calcula `z` y la activación de cada neurona, capa por capa, hasta la predicción | las activaciones intermedias de cada capa (hacen falta después) |
| **Backward** | salida → entrada | calcula el error en la salida y lo propaga hacia atrás, capa por capa, usando la regla de la cadena y las activaciones guardadas en el forward | el gradiente de la pérdida respecto de cada peso y cada sesgo |

```mermaid
flowchart LR
    subgraph Forward
        direction LR
        I[Entrada] --> H1[Capa oculta 1]
        H1 --> H2[Capa oculta 2]
        H2 --> O[Salida]
    end
    subgraph Backward
        direction RL
        O2[Error en salida] --> G2[Gradiente capa oculta 2]
        G2 --> G1[Gradiente capa oculta 1]
        G1 --> GE[Gradiente capa entrada]
    end
    O -.-> O2
```

**La derivación del curso, sobre una red concreta.** El material desarrolla el cálculo sobre una red de 2 entradas (`x₁`, `x₂`), 2 neuronas ocultas (`O₁`, `O₂`) y 1 salida (`S`), con **sigmoide** en todas las activaciones y **error cuadrático** como pérdida. De esos dos supuestos salen los dos factores que se repiten en todas las fórmulas: la derivada del error, `(ŷ − y)`, y la de la sigmoide, `ŷ(1 − ŷ)`.

Para un peso de la **capa de salida**, la cadena tiene tres factores:

```
∂E/∂p₂₁ = (∂E/∂ŷ) · (∂ŷ/∂sumaₛ) · (∂sumaₛ/∂p₂₁) = (ŷ − y) · ŷ(1 − ŷ) · salida₀₁ = δₛ · salida₀₁
```

Los dos primeros factores se agrupan bajo el nombre **`δₛ`**, y ese agrupamiento es el corazón del algoritmo: sirve para todos los parámetros de esa capa (`∂E/∂p₂₂ = δₛ·salida₀₂`, `∂E/∂sesgo₂₁ = δₛ·1`) y se **reutiliza** para las capas de más atrás. El sesgo se deriva como un peso cuya entrada vale siempre 1.

Para un peso de la **capa oculta**, la cadena se alarga a cinco factores:

```
∂E/∂p₁₁ = (ŷ − y) · ŷ(1 − ŷ) · p₂₁ · salida₀₁(1 − salida₀₁) · x₁
           └────── δₛ ──────┘   └── cruza el peso ──┘  └─ activación de O₁ ─┘  └ entrada
```

Leída de izquierda a derecha, la fórmula **es** el recorrido del error hacia atrás: sale del error, atraviesa la activación de salida, cruza el peso `p₂₁` hacia la capa oculta, atraviesa la activación de `O₁` y termina en la entrada `x₁`. Los demás pesos siguen el mismo patrón cambiando qué neurona oculta y qué entrada intervienen.

Con los gradientes calculados, cada parámetro se actualiza con la regla del descenso de gradiente (cap. 35):

```
p₁₁^nuevo = p₁₁^viejo − η · (∂E/∂p₁₁)
```

> **De dónde sale el 0,25.** La derivada de la sigmoide, `salida(1 − salida)`, vale **como máximo 0,25** (en `salida = 0,5`). Cada capa que el error atraviesa hacia atrás multiplica por un factor de ese tipo, así que el gradiente se achica al menos a la cuarta parte por capa aunque ninguna neurona esté saturada. Ese es el mecanismo exacto del gradiente desvaneciente.

**Por qué es eficiente.** La alternativa ingenua sería derivar la pérdida respecto de cada peso por separado, desde cero, recorriendo toda la red cada vez. Backpropagation evita ese trabajo repetido: calcula el gradiente de la última capa una sola vez y lo **reutiliza** para calcular el de la capa anterior, y así sucesivamente. Cada capa reaprovecha el resultado ya calculado de la capa siguiente en lugar de recomputar la cadena completa desde la salida. Esa reutilización es lo que hace viable entrenar redes de muchas capas: el costo crece linealmente con la cantidad de capas, no exponencialmente.

**Por qué la saturación lo rompe.** El gradiente que llega a una capa temprana es un **producto** de todas las derivadas locales de las capas posteriores (esa es justamente la regla de la cadena). Si la activación usada es sigmoide o tanh (cap. 32), su derivada es casi 0 en los extremos donde la neurona satura. Multiplicar varios números casi nulos entre sí da un número todavía más chico: el gradiente que llega a las primeras capas se **desvanece**, y esas capas dejan de actualizarse aunque el error en la salida siga siendo grande. Es la misma razón por la que ReLU —que no satura del lado positivo— se volvió la activación por defecto en redes con varias capas ocultas.


### 38. Persistencia de modelos

Entrenar es caro; predecir es barato. **Persistir** un modelo es guardarlo entrenado en disco para poder usarlo después sin volver a entrenarlo. Lo que la slide destaca:

- **Evita reentrenar** cada vez, ahorrando tiempo y recursos.
- Permite **servirlo desde un servidor**, procesando datos en tiempo real.
- Permite **desplegar varias instancias** en distintos nodos, bajando la latencia.
- Permite **versionar** el modelo y **revertir** si una actualización rompe algo.

**Las dos librerías.**

| | `joblib` | `pickle` |
|---|---|---|
| Origen | externa (viene con scikit-learn) | estándar de Python |
| Fuerte en | **objetos grandes** con arrays de NumPy | objetos Python en general |
| Compresión | sí, `compress=0..9` | no directamente |
| Recomendada para sklearn | **sí** | funciona, pero es la segunda opción |

```python
import joblib
joblib.dump(mlp, 'mi_modelo.joblib')                        # guardar
mlp = joblib.load('mi_modelo.joblib')                       # cargar
joblib.dump(mlp, 'comprimido.joblib', compress=3)           # guardar comprimido

import pickle
with open('mi_modelo.pkl', 'wb') as f:                      # 'wb' = escritura binaria
    pickle.dump(mlp, f)
with open('mi_modelo.pkl', 'rb') as f:                      # 'rb' = lectura binaria
    mlp = pickle.load(f)
```

> **Nota (medido con el modelo del notebook de clase):** el `.joblib` sin comprimir pesa 392.276 bytes y el comprimido 375.378 — un **4% menos**. Con un MLP chico la compresión casi no aporta y agrega tiempo de guardado y carga; rinde en modelos grandes.

**Tres cosas que el material no menciona y que importan más que la sintaxis.**

- **Seguridad.** Deserializar un pickle **ejecuta código arbitrario**: un archivo malicioso corre lo que quiera al abrirse. Nunca cargar un modelo de origen desconocido. `joblib` usa pickle por debajo, así que hereda el riesgo. Para modelos que viajan entre organizaciones existen formatos de intercambio como **ONNX** o **PMML**, que describen el modelo sin serializar objetos de Python.
- **Compatibilidad de versiones.** Un modelo guardado con una versión de scikit-learn puede fallar al cargarse con otra o, peor, cargar sin error y comportarse distinto. Guardar junto al modelo la versión de scikit-learn, de Python y de las dependencias.
- **Guardar el `Pipeline`, no el estimador suelto.** `joblib.dump(mlp, ...)` serializa solo los pesos e hiperparámetros del estimador: **no** guarda el `StandardScaler` (cap. 5). Si el modelo se entrenó con datos escalados y al cargarlo recibe datos crudos, las predicciones salen mal **sin ningún error visible**.

> **Nota (verificado, Iris con 2 atributos, `sklearn 1.9.0`):** el mismo `MLPClassifier` cargado desde disco da **96,67%** de exactitud con los datos escalados y **36,67%** con los datos crudos — peor que predecir la clase mayoritaria. No se lanza ninguna excepción: el modelo acepta la entrada, la procesa y devuelve predicciones con toda normalidad. Guardado como `Pipeline`, en cambio, recibe los datos crudos y devuelve el 96,67% correcto. Es un error que solo se detecta comparando métricas contra las del entrenamiento.

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), MLPClassifier(random_state=42)).fit(X_train, y_train)
joblib.dump(pipe, 'modelo_completo.joblib')   # el escalado viaja con la red
```

Del mismo modo, el **orden de los atributos** al predecir debe ser el mismo del entrenamiento: el modelo recibe posiciones, no nombres de columna. Y la entrada va como matriz 2D — de ahí el `.reshape(1, -1)` para predecir sobre una sola muestra.

## Parte VIII — Referencia técnica

Las partes anteriores explican **qué es** cada técnica y **por qué** funciona. Esta es la parte de consulta: para cada familia de modelos, cuándo aplicarla, sus hiperparámetros, cómo se evalúa y el snippet de `scikit-learn` listo para usar. No se lee de corrido.

### Modelos clásicos de clasificación

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

### Regresión lineal

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

### Redes neuronales

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

---

## Desafíos profesionales

Los módulos 02, 05 y 09 del programa no son teóricos: son proyectos integradores end-to-end. Cada uno tiene sus apuntes propios en la carpeta del módulo.

### Etapa 1 — Exploración y limpieza

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
