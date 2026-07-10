# Apuntes — Módulo 03: Modelado avanzado en Machine Learning

> Apuntes sintetizados de la teoría del módulo. Repaso global del programa en [`../APUNTES-DATA-SCIENCE.md`](../APUNTES-DATA-SCIENCE.md).

## 1. Regresión lineal

Predice una respuesta **cuantitativa** `Y` asumiendo relación aproximadamente lineal.

- **Simple:** `Y = β₀ + β₁·X + ε`
- **Múltiple:** `Y = β₀ + β₁X₁ + … + βₚXₚ + ε`

`β₀` = intercepto, `βⱼ` = pendientes, `ε` = error. Entrenar = **estimar los coeficientes** (β̂).

- **Residuo:** `eᵢ = yᵢ − ŷᵢ`.
- **Mínimos cuadrados (OLS):** minimiza `RSS = Σ(yᵢ − ŷᵢ)²`.

## 2. Función de costo y mínimos cuadrados

La **función de costo** mide la discrepancia entre las predicciones del modelo y los valores reales; entrenar = **minimizarla**. En regresión lineal la función de costo es `RSS` (suma de residuos al cuadrado).

- **Solución cerrada (ecuación normal):** `β̂ = (Xᵀ·X)⁻¹·Xᵀ·y`.
- Pasos: (1) calcular predicciones, (2) calcular residuos `eᵢ = yᵢ − ŷᵢ`, (3) minimizar `Σeᵢ²`.

## 3. Métricas de regresión

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| MAE | `(1/n)·Σ\|yᵢ − ŷᵢ\|` | Error medio absoluto; misma unidad que `y`; robusto a outliers. |
| MSE | `(1/n)·Σ(yᵢ − ŷᵢ)²` | Penaliza más los errores grandes. |
| RMSE | `√MSE` | En la unidad de `y`; más interpretable. |
| R² | `1 − RSS/TSS` | Proporción de varianza explicada. |

**Descomposición:** `TSS = ESS + RSS` (Total = Explicada + Residual).

**R²:** `1` = ajuste perfecto · `0` = no mejor que la media · `< 0` = peor que la media.

## 4. Validación cruzada

Estima la **capacidad de generalización** a datos no vistos y ayuda a prevenir el sobreajuste; más fiable que evaluar sobre los datos de entrenamiento.

- **K-Fold:** se parte el dataset en `k` folds; se entrena en `k-1` y se valida en el restante, rotando y promediando la métrica sobre los `k` folds.
- **LOOCV:** caso extremo con `k = n` (una observación por fold); muy costoso.
- **Stratified K-Fold:** preserva la proporción de clases en cada fold (clasificación con clases desbalanceadas).
- En regresión se promedian MSE / RMSE / R² entre folds.

## 5. Inferencia sobre los coeficientes

**Significación global (test F):** `H₀: β₁ = … = βₚ = 0` vs `H₁: al menos un βⱼ ≠ 0`, con `F = ((TSS − RSS)/p) / (RSS/(n − p − 1))`. Un p-value chico (`Prob (F-statistic)`) indica que el conjunto de predictores explica `Y`.

Test de significación individual para `βⱼ`:

- **H₀:** `βⱼ = 0` (no hay relación). **H₁:** `βⱼ ≠ 0` (hay relación).
- Si `β₁ = 0`, el modelo se reduce a `Y = β₀ + ε` y `X` no aporta.
- **p-value chico** (< 0,05) → se rechaza H₀: hay evidencia de relación. Complementar con **intervalos de confianza** de los coeficientes.

## 6. Dilema sesgo-varianza

El error de un modelo se descompone en **sesgo² + varianza + error irreducible**:

- **Sesgo (bias):** error sistemático por un modelo demasiado **simple** (pocos grados de libertad). `E[f̂(x)] ≠ E[f(x)]`. No baja por más datos que haya → **underfitting**.
- **Varianza:** sensibilidad a la muestra de un modelo demasiado **complejo** (muchos grados de libertad); ajusta regularidades espurias → **overfitting**.
- **Trade-off:** al subir la complejidad, el sesgo baja y la varianza sube. El **error total tiene forma de U**; el mínimo es la **complejidad óptima** (ni muy simple ni muy complejo).

## 7. Regularización (Ridge, Lasso, Elastic Net)

Agrega una penalidad a la función de costo para achicar los coeficientes y controlar el sobreajuste: `CF = RSS + α · penalidad(β)`. El hiperparámetro `α` (la `λ`) regula la fuerza y se elige por **cross-validation**. **Estandarizar** los regresores antes, porque la penalidad depende de la escala.

| Método | Penalidad | Norma | Efecto |
|--------|-----------|-------|--------|
| **Ridge** | `λ Σ βⱼ²` | L2 | Achica los β hacia 0, pero ninguno llega a 0 exacto; reparte peso entre variables colineales. |
| **Lasso** | `λ Σ \|βⱼ\|` | L1 | Lleva β **exactamente a 0** → selección de variables (modelos dispersos). |
| **Elastic Net** | `λ(‖β‖₁ + α‖β‖₂²)` | L1 + L2 | Combina ambas; `α` regula el peso Lasso vs. Ridge. Calibra **dos** hiperparámetros. |

## 8. Modelos de ensamble

Combinan varios modelos para mejorar el rendimiento. Necesitan modelos **poco correlacionados** entre sí para aportar información. Dos familias:

### Averaging (paralelo) — reduce **varianza**

Entrena estimadores independientes y promedia/vota. Los modelos base suelen ser de alta varianza (árboles sin podar).

- **Bagging** (Bootstrap Aggregation): entrena N modelos sobre N datasets de *bootstrap* (muestreo con reemplazo) y agrega. Si los árboles tienen varianza `S²`, el ensamble tiende a `S²/N`. En clasificación vota por mayoría; en regresión promedia.
- **Random Forest:** bagging de árboles + en cada nodo se considera solo un subconjunto aleatorio de `M` features (descorrelaciona). Reglas empíricas: `M = P/3` (regresión), `M = √P` (clasificación). Bagging es el caso `M = P`.
- **Extra Trees:** además elige los cortes al azar (más aleatoriedad).

### Boosting (secuencial) — reduce **sesgo**

Entrena modelos en secuencia; cada uno aprende de los errores del anterior. Combina modelos **débiles** en uno fuerte.

| Método | Idea | sklearn |
|--------|------|---------|
| **AdaBoost** | Sube el peso de las observaciones mal clasificadas; usa *stumps* (árbol de 1 corte) con voto ponderado según su error. | `AdaBoostClassifier/Regressor` |
| **Gradient Boosting** | Cada árbol predice los **residuos** del anterior; `ŷ = y₁ + η·r₁ + … + η·r_N`. `η` (learning rate) controla la convergencia (η↓ ⇒ N↑). | `GradientBoostingClassifier/Regressor` |
| **XGBoost** | Gradient boosting optimizado: usa **gradientes de segundo orden**, es rápido, paraleliza, maneja nulos y hace pruning. | `xgboost` |

## 9. SVM (Support Vector Machines)

Algoritmo supervisado (clasificación y regresión) que busca el **hiperplano que mejor separa las clases** maximizando el **margen** (distancia a los puntos más cercanos, los *vectores de soporte*). Más margen ⇒ mejor generalización.

- **Función de costo:** `L(w,b) = ½‖w‖² + C·Σ max(0, 1 − yᵢ(w·xᵢ + b))`. El término `½‖w‖²` maximiza el margen; `C` regula el trade-off entre margen y errores. Se resuelve como optimización cuadrática (multiplicadores de Lagrange).
- **SVM lineal:** hiperplano lineal; sirve cuando las clases son linealmente separables.
- **SVM no lineal (kernel trick):** un **kernel** calcula el producto escalar en un espacio de mayor dimensión **sin transformar explícitamente los datos**, permitiendo fronteras no lineales. Kernels comunes:
  - Lineal: `K = xᵢ·xⱼ` (datos linealmente separables).
  - Polinomial: `K = (xᵢ·xⱼ + c)^d` (interacciones polinómicas).
  - RBF: `K = exp(−γ‖xᵢ−xⱼ‖²)` (relaciones complejas; `γ` define el alcance).
  - Sigmoidal: `K = tanh(α·xᵢ·xⱼ + c)` (similar a redes neuronales).
- Los kernels también se usan fuera de SVM: **SVR** (regresión) y **KPCA** (reducción de dimensionalidad no lineal). Contra: elegir kernel/parámetros es delicado y es costoso en datasets grandes.
- **Escala:** estandarizar siempre (SVM trabaja con distancias). En sklearn: `SVC` / `SVR`.

### Hiperparámetros clave en SVM
- **C:** penalización de los errores. `C` alto ⇒ intenta clasificar/ajustar todo (riesgo de overfitting); `C` bajo ⇒ más margen y regularización (riesgo de underfitting).
- **γ (gamma, kernel RBF):** influencia de cada observación. `γ` alto ⇒ función muy flexible (overfitting).
- **kernel:** lineal / polinomial / RBF / sigmoidal. Cambia drásticamente el resultado (p. ej. en SVR sobre Hitters, RBF ≈ 0.65 de R² vs. sigmoidal negativo).

## 10. Ajuste de hiperparámetros (tuning)

Buscar la combinación de hiperparámetros que mejor generaliza, balanceando **overfitting** y **underfitting**. Se evalúa con validación cruzada.

| Método | Idea | Pros / Contras | sklearn |
|--------|------|----------------|---------|
| **Grid Search** | Prueba **todas** las combinaciones de una grilla. | Exhaustivo y simple / muy costoso, no escala. | `GridSearchCV` |
| **Random Search** | Muestrea combinaciones **al azar**. | Rápido, buena cobertura / no garantiza el óptimo. | `RandomizedSearchCV` |
| **Bayesian Optimization** | Modelo **probabilístico** que usa iteraciones previas para elegir las próximas. | Eficiente en evaluaciones, dirigido / más complejo. | `optuna`, `skopt` |

**Optimización convexa vs no convexa** (de la función objetivo del modelo, no de los hiperparámetros):
- **Convexa:** un único **mínimo global**, fácil y garantizado de alcanzar. Ej.: regresión lineal, SVM lineal.
- **No convexa:** múltiples **mínimos locales**; más flexible pero difícil de optimizar (heurísticas). Ej.: redes neuronales profundas, SVM no lineal.

## 11. Descenso del gradiente

Algoritmo de optimización **iterativo** para minimizar la función de costo cuando no hay solución analítica (regresión logística, redes neuronales). Se arranca en un punto y se avanza en la **dirección negativa del gradiente**:

`βⱼ ← βⱼ − α · ∂J/∂βⱼ`

- **Gradiente:** vector de derivadas; apunta al mayor aumento y vale 0 en un mínimo/máximo.
- **Learning rate (α):** hiperparámetro clave. Muy grande ⇒ oscila y no converge; muy chico ⇒ converge lento. Conviene **normalizar** las variables para acelerar la convergencia.
- **Convexidad:** si la función de costo es convexa (regresión lineal) converge al **mínimo global**; si no (logística, redes), conviene reiniciar desde distintos puntos para no quedar en un mínimo local.

## 12. Buenas prácticas del módulo

- Verificar supuestos de la regresión lineal (linealidad, homocedasticidad, normalidad de residuos, independencia).
- Estandarizar variables cuando se comparan coeficientes o se usa regularización.
- Comparar modelos con métricas en conjunto de test, no de entrenamiento.
