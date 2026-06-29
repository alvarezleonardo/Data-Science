<!-- Convertido automáticamente desde: Regularización - Ridge y Lasso.pdf -->

Regularización: Ridge y Lasso

Función de pérdida de una regresión lineal:

    CF = Σ (ŷᵢ − yᵢ)²

Las técnicas de regularización agregan una "penalidad" a esta función de costo:

    CF = Σ (ŷᵢ − yᵢ)² + α · θ⃗

`θ` (theta) es el vector de parámetros del modelo (en regresión lineal, los
betas) y `alpha` regula la fuerza de la penalización: cuanto más grande, mayor
es la penalización.

Normas L1 y L2

- Norma L0: la cantidad de elementos distintos de cero en el vector.
- Norma L1: la suma de los valores absolutos de los elementos. `||β||₁ = Σ |xᵢ|`
- Norma L2: la raíz cuadrada de la suma de cuadrados. `||β||₂ = √(Σ xᵢ²)`

Normalización

- Normalización L1: que la suma del valor absoluto sea unitaria.
  `Σⱼ ||xᵢⱼ|| = 1 ∀ i`
- Normalización L2: que la suma del valor absoluto al cuadrado sea unitaria.
  `Σⱼ ||xᵢⱼ||² = 1 ∀ i`

01 Regresión Ridge

    Σᵢ (yᵢ − β₀ − Σⱼ βⱼ xᵢⱼ)² + λ Σⱼ βⱼ² = RSS + λ Σⱼ βⱼ²

Existe un término de penalización que es menor cuando los betas se acercan a
cero; por lo tanto tiene el efecto de achicar los coeficientes hacia cero (tanto
si son negativos como positivos). El hiperparámetro `λ` maneja la ponderación de
cada término.

¿Cuál es el mejor valor para lambda? Como siempre, lo elegimos mediante **cross
validation**.

Escala: en Ridge tanto la estimación de los coeficientes como la predicción son
sensibles a la escala. Si una variable está en una escala que le da un valor
absoluto mayor, esto afecta el cálculo de la suma de cuadrados del vector de
coeficientes. Por eso es importante **estandarizar** (dividir por el desvío
estándar) todos los regresores antes de ejecutar una regresión Ridge.

    x̃ᵢⱼ = xᵢⱼ / √( (1/n) Σᵢ (xᵢⱼ − x̄ⱼ)² )

02 Regresión Lasso

La regresión Ridge tiene una clara desventaja: incluye todos los `p` predictores
en el modelo final, a diferencia de los modelos que eligen un subconjunto de
predictores. Lasso es una alternativa que corrige esta desventaja.

Lasso utiliza la norma **L1** en la penalización, a diferencia de Ridge que usa
la **L2**:

    L(β) = Σᵢ (yᵢ − ŷᵢ)² + λ Σⱼ |βⱼ|

Como en Ridge, Lasso "achica" los coeficientes estimados. Pero la regularización
L1 fuerza coeficientes a valer **exactamente cero** cuando `λ` es lo
suficientemente grande. Por lo tanto, Lasso hace **selección de variables**:
genera modelos dispersos (sparse). Al igual que en Ridge, la elección de un buen
`λ` es crítica y se hace por cross-validation.

03 Elastic Net

    λ₁·||β||₁ + λ₂·||β||₂² = λ·( ||β||₁ + α·||β||₂² )

ElasticNet combina linealmente lo mejor de ambos mundos. El parámetro `λ` regula
la complejidad del modelo; el parámetro `α` regula la importancia relativa de
Lasso vs. Ridge. Permite obtener soluciones parsimoniosas y bien condicionadas.

No free lunch: ahora hay que calibrar dos hiperparámetros.

¡Muchas gracias!
