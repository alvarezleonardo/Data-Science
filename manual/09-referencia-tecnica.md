[Índice](README.md) · [← Anterior](08-deep-learning-con-frameworks.md)

# Parte VIII — Referencia técnica

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
