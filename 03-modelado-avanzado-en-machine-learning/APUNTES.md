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

## 8. Buenas prácticas del módulo

- Verificar supuestos de la regresión lineal (linealidad, homocedasticidad, normalidad de residuos, independencia).
- Estandarizar variables cuando se comparan coeficientes o se usa regularización.
- Comparar modelos con métricas en conjunto de test, no de entrenamiento.
