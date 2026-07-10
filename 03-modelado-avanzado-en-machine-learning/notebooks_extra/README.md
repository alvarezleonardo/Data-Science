# Notebooks extra — Módulo 03

Notebooks de práctica adicional, **no evaluados**, que integran los temas del módulo.

## `Ejercicio_Incremental.ipynb`

Ejercicio integrador sobre el dataset **California housing** (`../datasets/housing.csv`), prediciendo `median_house_value`. Recorre el flujo completo de un problema de regresión:

1. **EDA** — estructura, tipos, distribución geográfica (lat/lon), correlación ingreso/precio.
2. **Valores faltantes** — imputación de `total_bedrooms` con la media.
3. **Outliers** — recorte por percentil 0.9 de las variables continuas.
4. **Variables y distribuciones** — `ocean_proximity`, rangos de ingreso y de valor.
5. **Preparación** — selección de features, train/test split y escalado (`MinMaxScaler`).
6. **Progresión de modelos**, evaluando R² / MSE / RMSE en cada uno:
   - Regresión lineal (base)
   - Árbol de decisión (variando `max_depth`)
   - SVR (kernels rbf/sigmoid/poly, `C`)
   - Random Forest (`n_estimators`, `max_depth`, `max_features`)
   - XGBoost (`n_estimators`, `max_depth`, `learning_rate`)
7. **Validación cruzada** (10 folds) comparando XGBoost vs Random Forest.
8. **Ajuste de hiperparámetros** — `GridSearchCV` y `RandomizedSearchCV`.

### Correcciones aplicadas

Al revisarlo se corrigieron errores para que corra en versiones actuales de las librerías (pandas 3.x, scikit-learn 1.9):

- **Imputación de nulos:** `df['total_bedrooms'].fillna(..., inplace=True)` fallaba con `ChainedAssignmentError` en pandas 3 → se reemplazó por asignación directa (`df['total_bedrooms'] = df['total_bedrooms'].fillna(...)`).
- **Countplot de `ocean_proximity`:** `ocean_values[i]` indexaba por etiqueta y daba `KeyError` → se cambió por `ocean_values.iloc[i]` (posicional).
- **Variantes de Random Forest:** un typo (`mode=` en vez de `model=`) hacía que 3 configuraciones no se aplicaran (se reentrenaba el modelo por defecto) → corregido a `model=`.

Verificado: corre de punta a punta sin errores (las secciones de GridSearch son sólo lentas, no fallan).

## `evaluacion_final_modulo.ipynb`

Evaluación final del módulo sobre **California housing** (`../datasets/housing.csv`). Consignas en [`evaluacion_final_modulo.md`](evaluacion_final_modulo.md).

Deja armado el **preprocesamiento pedido** (imputación de nulos con la media, `StandardScaler` en las features numéricas, `OneHotEncoder` en `ocean_proximity`, unificación en un dataframe nuevo) y el **split** (20% test, `random_state=42`), listo para resolver el cuestionario (modelos y evaluación) que se irá agregando.

[← Volver al módulo](../README.md)
