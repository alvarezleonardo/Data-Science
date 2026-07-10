# 03 — Modelado avanzado en Machine Learning

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 7 | 28 | 🔄 En curso (71% · 5/7) |

Profundización en modelos de regresión y técnicas avanzadas de modelado supervisado.

Apuntes teóricos de este módulo: **[APUNTES.md](APUNTES.md)** · repaso global: [../APUNTES-DATA-SCIENCE.md](../APUNTES-DATA-SCIENCE.md#módulo-2--modelado-avanzado).

## Contenido

### `teoria/`
Slides del curso (PDF + conversión `.md`):

| Tema | PDF | Markdown |
|------|-----|----------|
| Introducción al Aprendizaje Supervisado | [PDF](<teoria/Introducción al Aprendizaje Supervisado.pdf>) | [MD](<teoria/Introducción al Aprendizaje Supervisado.md>) |
| Principios de Regresión Lineal | [PDF](<teoria/Principios de Regresión Lienal.pdf>) | [MD](<teoria/Principios de Regresión Lienal.md>) |
| Modelos de Regresión | [PDF](<teoria/Modelos de Regresión.pdf>) | [MD](<teoria/Modelos de Regresión.md>) |
| Funciones de Costo y Mínimos Cuadrados | [PDF](<teoria/Funciones de Costo.pdf>) | [MD](<teoria/Funciones de Costo.md>) |
| Regresión Lineal Múltiple | [PDF](<teoria/Regresión Lineal Múltiple.pdf>) | [MD](<teoria/Regresión Lineal Múltiple.md>) |
| Validación Cruzada | [PDF](<teoria/Validación Cruzada.pdf>) | [MD](<teoria/Validación Cruzada.md>) |
| Dilema Sesgo - Varianza | [PDF](<teoria/Dilema Sesgo - Varianza.pdf>) | [MD](<teoria/Dilema Sesgo - Varianza.md>) |
| Regularización: Ridge y Lasso | [PDF](<teoria/Regularización - Ridge y Lasso.pdf>) | [MD](<teoria/Regularización - Ridge y Lasso.md>) |
| Introducción a Modelos de Ensamble | [PDF](<teoria/Introducción a Modelos de Ensamble.pdf>) | [MD](<teoria/Introducción a Modelos de Ensamble.md>) |
| Introducción a Random Forest | [PDF](<teoria/Introducción a Random Forest.pdf>) | [MD](<teoria/Introducción a Random Forest.md>) |
| Introducción a Boosting | [PDF](<teoria/Introducción a Boosting.pdf>) | [MD](<teoria/Introducción a Boosting.md>) |
| Boosting - ADA Boosting | [PDF](<teoria/Introducción a Boosting - ADA Boosting.pdf>) | [MD](<teoria/Introducción a Boosting - ADA Boosting.md>) |
| Boosting - Gradient Boosting | [PDF](<teoria/Introducción a Boosting - Gradient Boosting.pdf>) | [MD](<teoria/Introducción a Boosting - Gradient Boosting.md>) |
| Boosting - XGBoost | [PDF](<teoria/Introducción a Boosting - XGBoost.pdf>) | [MD](<teoria/Introducción a Boosting - XGBoost.md>) |
| Fundamentos de SVM | [PDF](<teoria/Fundamentos de SVM.pdf>) | [MD](<teoria/Fundamentos de SVM.md>) |
| Introducción a los Kernels | [PDF](<teoria/Introducción a los Kernels.pdf>) | [MD](<teoria/Introducción a los Kernels.md>) |
| Técnicas de Optimización de SVM e Hiperparámetros | [PDF](<teoria/Técnicas de Optimización de SVM y Ajuste de Hiperparámetros.pdf>) | [MD](<teoria/Técnicas de Optimización de SVM y Ajuste de Hiperparámetros.md>) |
| Métodos de Optimización (convexa vs no convexa) | [PDF](<teoria/Métodos de optimización.pdf>) | [MD](<teoria/Métodos de optimización.md>) |
| Introducción al Descenso del Gradiente | [PDF](<teoria/Introducción al Descenso del Gradiente.pdf>) | [MD](<teoria/Introducción al Descenso del Gradiente.md>) |
| Optimización de Hiperparámetros | [PDF](<teoria/Optimización de Hiperparámetros.pdf>) | [MD](<teoria/Optimización de Hiperparámetros.md>) |
| Más Práctica 1 (ejercicio — Bikes) | [PDF](<teoria/Más Práctica 1.pdf>) | [MD](<teoria/Más Práctica 1.md>) |
| Recursos del Curso | [PDF](<teoria/OD_ML2_ESP_M02_S02_Recursos del Curso .pdf>) | [MD](<teoria/OD_ML2_ESP_M02_S02_Recursos del Curso .md>) |

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

[← Volver al índice](../README.md)
