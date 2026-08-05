# Selección de Variables con Regularización

> Conversión a Markdown de la slide del curso (Clase 19, Módulo 4). El PDF original está al lado.

## 1. Idea general

La función de pérdida (costo) de una regresión lineal es la suma de errores al cuadrado:

```
CF = Σ (ŷ_i - y_i)²
```

Las técnicas de **regularización** agregan una **penalidad** a esa función de costo:

```
CF = Σ (ŷ_i - y_i)² + α · θ⃗
```

- **θ (theta)** es el vector de parámetros del modelo (en regresión lineal, los **betas**).
- **α (alpha)** regula la **fuerza de la penalización**: cuanto más grande, mayor la penalización.

## 2. Normas de un vector

- **Norma L0:** cantidad de elementos distintos de cero en el vector.
- **Norma L1:** suma de los **valores absolutos** de los elementos → `‖β‖₁ = Σ |x_i|`.
- **Norma L2:** raíz cuadrada de la **suma de cuadrados** de los elementos → `‖β‖₂ = √(Σ x_i²)`.

## 3. Regresión Ridge (norma L2)

Agrega a la RSS un término de penalización proporcional a la **suma de los cuadrados** de los coeficientes:

```
Σ (y_i - β₀ - Σ β_j x_ij)² + λ Σ β_j²  =  RSS + λ Σ β_j²
```

- El término de penalización es menor cuando los betas se acercan a cero → **achica** los coeficientes hacia cero (positivos o negativos), pero **nunca exactamente a cero**.
- El hiperparámetro **λ (lambda)** maneja la ponderación:
  - **λ muy grande:** reduce demasiado los coeficientes (underfitting).
  - **λ muy pequeño:** apenas los penaliza (cercano a OLS).
- **¿Cómo elegir λ óptimo?** Con **Cross Validation**: se prueban distintos valores de λ y se elige el de mejor rendimiento en datos no vistos.
- **Sensible a la escala:** la estimación y la predicción dependen de la escala. Es importante **estandarizar** (dividir por el desvío estándar) todos los regresores antes de ejecutar Ridge, para que estén en unidades de su propio desvío y no en unidades físicas.

## 4. Regresión Lasso (norma L1)

Lasso corrige una **desventaja de Ridge**: Ridge incluye **todos** los predictores `p` en el modelo final, mientras que Lasso puede **descartar** predictores.

```
L(β) = Σ (y_i - ŷ_i)² + λ Σ |β_j|
```

- Usa la **norma L1** en la penalización (Ridge usa **L2**).
- La regularización L1 fuerza algunos coeficientes a valer **exactamente cero** cuando λ es lo suficientemente grande → **hace selección de variables**.
- Genera **modelos dispersos** (sparse): modelos con una selección de variables incorporada.
- Al igual que en Ridge, la elección de un buen λ es crítica → nuevamente **cross-validation** es el método para elegirlo.

## 5. Elastic Net (L1 + L2)

Combina linealmente **lo mejor de ambos mundos** (Lasso + Ridge):

```
λ₁·‖β‖₁ + λ₂·‖β‖₂²  =  λ · (‖β‖₁ + α·‖β‖₂²)
```

- El parámetro **λ** regula la **complejidad** del modelo.
- El parámetro **α** regula la **importancia relativa de Lasso vs. Ridge**.
- Permite obtener soluciones **parsimoniosas y bien condicionadas** (selección de variables + estabilidad ante colinealidad).
- **No free lunch:** ahora hay que **calibrar dos hiperparámetros**.

## Resumen

| Método | Norma | ¿Anula coeficientes? | Uso principal |
|--------|:-----:|:--------------------:|---------------|
| **Ridge** | L2 | No (los achica) | Colinealidad; conservar todos los predictores |
| **Lasso** | L1 | **Sí** (a cero exacto) | Selección de variables / modelos dispersos |
| **Elastic Net** | L1 + L2 | Sí (parcial) | Combina selección y estabilidad; 2 hiperparámetros |

> Práctica asociada: [notebooks/seleccion_variables_embedded_practica.ipynb](../notebooks/seleccion_variables_embedded_practica.ipynb) — métodos embedded de selección de variables. Enlaza con [Criterios de Selección de Modelos](<Criterios de Selección de Modelos.md>) (Clase 18) y [Comprender la Maldición de la Dimensión](<Comprender la Maldición de la Dimensión.md>) (Clase 16).
