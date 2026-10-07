# 03 — Modelado avanzado en Machine Learning

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 28 | ✅ Aprobado (7/7) |

Profundización en modelos de regresión y técnicas avanzadas de modelado supervisado.

Apuntes teóricos de este módulo: **[APUNTES.md](APUNTES.md)** · guía de estudio para examen: **[guia-estudio.md](guia-estudio.md)** · repaso global: [../manual/03-modelos-lineales.md](../manual/03-modelos-lineales.md#parte-iii--modelos-lineales).

## Contenido

> Los **23 PDF originales** del curso están en [`teoria/material/`](teoria/material/); en `teoria/` quedan sus conversiones a Markdown.

### `teoria/`
Slides del curso (PDF + conversión `.md`):

| Tema | PDF | Markdown |
|------|-----|----------|
| Introducción al Aprendizaje Supervisado | [PDF](<teoria/material/introduccion-al-aprendizaje-supervisado.pdf>) | [MD](<teoria/introduccion-al-aprendizaje-supervisado.md>) |
| Principios de Regresión Lineal | [PDF](<teoria/material/principios-de-regresion-lienal.pdf>) | [MD](<teoria/principios-de-regresion-lienal.md>) |
| Modelos de Regresión | [PDF](<teoria/material/modelos-de-regresion.pdf>) | [MD](<teoria/modelos-de-regresion.md>) |
| Funciones de Costo y Mínimos Cuadrados | [PDF](<teoria/material/funciones-de-costo.pdf>) | [MD](<teoria/funciones-de-costo.md>) |
| Regresión Lineal Múltiple | [PDF](<teoria/material/regresion-lineal-multiple.pdf>) | [MD](<teoria/regresion-lineal-multiple.md>) |
| Validación Cruzada | [PDF](<teoria/material/validacion-cruzada.pdf>) | [MD](<teoria/validacion-cruzada.md>) |
| Dilema Sesgo - Varianza | [PDF](<teoria/material/dilema-sesgo-varianza.pdf>) | [MD](<teoria/dilema-sesgo-varianza.md>) |
| Regularización: Ridge y Lasso | [PDF](<teoria/material/regularizacion-ridge-y-lasso.pdf>) | [MD](<teoria/regularizacion-ridge-y-lasso.md>) |
| Introducción a Modelos de Ensamble | [PDF](<teoria/material/introduccion-a-modelos-de-ensamble.pdf>) | [MD](<teoria/introduccion-a-modelos-de-ensamble.md>) |
| Introducción a Random Forest | [PDF](<teoria/material/introduccion-a-random-forest.pdf>) | [MD](<teoria/introduccion-a-random-forest.md>) |
| Introducción a Boosting | [PDF](<teoria/material/introduccion-a-boosting.pdf>) | [MD](<teoria/introduccion-a-boosting.md>) |
| Boosting - ADA Boosting | [PDF](<teoria/material/introduccion-a-boosting-ada-boosting.pdf>) | [MD](<teoria/introduccion-a-boosting-ada-boosting.md>) |
| Boosting - Gradient Boosting | [PDF](<teoria/material/introduccion-a-boosting-gradient-boosting.pdf>) | [MD](<teoria/introduccion-a-boosting-gradient-boosting.md>) |
| Boosting - XGBoost | [PDF](<teoria/material/introduccion-a-boosting-xgboost.pdf>) | [MD](<teoria/introduccion-a-boosting-xgboost.md>) |
| Fundamentos de SVM | [PDF](<teoria/material/fundamentos-de-svm.pdf>) | [MD](<teoria/fundamentos-de-svm.md>) |
| Introducción a los Kernels | [PDF](<teoria/material/introduccion-a-los-kernels.pdf>) | [MD](<teoria/introduccion-a-los-kernels.md>) |
| Técnicas de Optimización de SVM e Hiperparámetros | [PDF](<teoria/material/tecnicas-de-optimizacion-de-svm-y-ajuste-de-hiperparametros.pdf>) | [MD](<teoria/tecnicas-de-optimizacion-de-svm-y-ajuste-de-hiperparametros.md>) |
| Métodos de Optimización (convexa vs no convexa) | [PDF](<teoria/material/metodos-de-optimizacion.pdf>) | [MD](<teoria/metodos-de-optimizacion.md>) |
| Introducción al Descenso del Gradiente | [PDF](<teoria/material/introduccion-al-descenso-del-gradiente.pdf>) | [MD](<teoria/introduccion-al-descenso-del-gradiente.md>) |
| Optimización de Hiperparámetros | [PDF](<teoria/material/optimizacion-de-hiperparametros.pdf>) | [MD](<teoria/optimizacion-de-hiperparametros.md>) |
| Más Práctica 1 (ejercicio — Bikes) | [PDF](<teoria/material/mas-practica-1.pdf>) | [MD](<teoria/mas-practica-1.md>) |
| Más Práctica 2 (ejercicio — Diamonds) | [PDF](<teoria/material/mas-practica-2.pdf>) | [MD](<teoria/mas-practica-2.md>) |
| Recursos del Curso | [PDF](<teoria/material/od-ml2-esp-m02-s02-recursos-del-curso.pdf>) | [MD](<teoria/od-ml2-esp-m02-s02-recursos-del-curso.md>) |

### `datasets/`
advertising, bikes, boston_data, Credit, diamonds, Hitters, housing, Movie_classification.

### `notebooks/`

| Notebook | Tema |
|----------|------|
| [regresion_lineal.ipynb](notebooks/regresion_lineal.ipynb) | Regresión lineal simple y múltiple con `statsmodels` (OLS), R² y validación cruzada (KFold). Dataset `advertising`. |
| [evaluacion_modelos_regresion.ipynb](notebooks/evaluacion_modelos_regresion.ipynb) | Regresión lineal con `scikit-learn`, métricas (MAE/MSE/RMSE/R²) y train/test split. Dataset `bikes`. |
| [regresion_polinomial.ipynb](notebooks/regresion_polinomial.ipynb) | Regresión polinómica con `PolynomialFeatures` + `Pipeline`; underfitting/overfitting y curva bias-variance (RMSE vs grado). Datos sintéticos. |
| [regularizacion_ridge_lasso.ipynb](notebooks/regularizacion_ridge_lasso.ipynb) | Regularización Ridge (L2) y Lasso (L1) con `RidgeCV`/`LassoCV`; estandarización, elección de `alpha` por CV y selección de variables ante colinealidad. Dataset `Credit`. |
| [ensambles_bagging_random_forest.ipynb](notebooks/ensambles_bagging_random_forest.ipynb) | Ensambles de averaging: `BaggingRegressor`, `RandomForestRegressor` y `ExtraTreesRegressor`; comparación de MSE vs. regresión lineal base. Dataset `Hitters`. |
| [adaboost.ipynb](notebooks/adaboost.ipynb) | Boosting con `AdaBoostRegressor` (base lineal) vs. regresión lineal. Dataset `Hitters`. |
| [gradient_boosting.ipynb](notebooks/gradient_boosting.ipynb) | Boosting con `GradientBoostingRegressor` (árboles); MSE vs. base lineal. Dataset `Hitters`. |
| [xgboost.ipynb](notebooks/xgboost.ipynb) | `XGBRegressor` sobre precios de viviendas; dummies para categóricas, nulos manejados por XGBoost, evaluación R²/MSE/RMSE. Dataset `housing` (California). |
| [svm.ipynb](notebooks/svm.ipynb) | Clasificación con `SVC` (kernel lineal): margen máximo, estandarización, accuracy + matriz de confusión. Dataset `iris`. |
| [svr.ipynb](notebooks/svr.ipynb) | Regresión con `SVR`: comparación de kernels (RBF/sigmoid/poly) y de `C`; R²/MSE/RMSE. Dataset `Hitters` (log Salary). |
| [descenso_gradiente.ipynb](notebooks/descenso_gradiente.ipynb) | Descenso del gradiente **desde cero** (clase propia) para regresión lineal; curva de pérdida y comparación con `LinearRegression`. Dataset `boston_data`. |
| [mas_practica_1_bikes.ipynb](notebooks/mas_practica_1_bikes.ipynb) | Resolución de *Más Práctica 1*: feature engineering de `hora` (numérica vs 23 dummies) y `día`; comparación de modelos por CV. `hora` categórica es la que más mejora (R² 0.33→0.61). Dataset `bikes`. |
| [mas-practica-2.ipynb](<notebooks/mas-practica-2.ipynb>) | Resolución de *Más Práctica 2*: regularización **Lasso** (`LassoCV` para elegir α) con `OneHotEncoder` + `MinMaxScaler`; comparación de coeficientes y métricas entre `statsmodels` y `scikit-learn`. Dataset `diamonds`. |

### `notebooks_extra/`
Práctica adicional no evaluada — ver [notebooks_extra/README.md](notebooks_extra/README.md).

| Notebook | Tema |
|----------|------|
| [ejercicio-incremental.ipynb](notebooks_extra/ejercicio-incremental.ipynb) | Ejercicio integrador sobre California `housing`: EDA + limpieza + progresión de modelos (LinearRegression → DecisionTree → SVR → RandomForest → XGBoost) + validación cruzada + `GridSearchCV`/`RandomizedSearchCV`. |
| [evaluacion_final_modulo.ipynb](notebooks_extra/evaluacion_final_modulo.ipynb) | Evaluación final del módulo sobre California `housing`: preprocesamiento (imputación + `StandardScaler` + `OneHotEncoder`) y cuestionario de 12 preguntas (árbol, regresión lineal, SVR, Random Forest y conceptuales). Consignas y respuestas en [evaluacion_final_modulo.md](notebooks_extra/evaluacion_final_modulo.md). |

[← Volver al índice](../README.md)
