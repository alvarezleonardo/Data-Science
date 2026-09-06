# Backpropagation (retropropagación)

> Conversión a Markdown de las slides del curso (unidad "Backpropagation y Gestión de modelos", bloque 01). El PDF original está al lado.

> **Nota:** el deck desarrolla el cálculo paso a paso con fórmulas en imagen, repitiendo la slide y agregando un término por página. Se consolidó cada derivación en su resultado final, transcribiendo las fórmulas.

## 1. Qué es

El **algoritmo de retropropagación (backpropagation)** es uno de los métodos más importantes para entrenar redes neuronales artificiales. Fue popularizado en la **década de 1980** y se utiliza para **minimizar la función de error** (o de pérdida) **ajustando los pesos** de la red.

## 2. La red del ejemplo

Todo el desarrollo usa una red concreta, de tres capas:

- **Entrada:** dos neuronas, `x₁` y `x₂`.
- **Oculta:** dos neuronas, `O₁` y `O₂`, cada una con su suma ponderada (`Σ`) y su activación **sigmoide** (`σ`).
- **Salida:** una neurona `S`, también con `Σ` y `σ`, que produce `ŷ`.

Los pesos se nombran `p` y los sesgos `sesgo`:

| Peso | Conecta |
|---|---|
| `p₁₁` | `x₁` → `O₁` |
| `p₁₂` | `x₁` → `O₂` |
| `p₁₃` | `x₂` → `O₁` |
| `p₁₄` | `x₂` → `O₂` |
| `p₂₁` | `O₁` → `S` |
| `p₂₂` | `O₂` → `S` |
| `sesgo₁₁`, `sesgo₁₂` | sesgos de `O₁` y `O₂` |
| `sesgo₂₁` | sesgo de `S` |

Dos supuestos del ejemplo: **la función de error es el error cuadrático** y **la función de salida es la sigmoide**. De ahí salen las dos derivadas que se repiten en todas las fórmulas: la del error, `(ŷ − y)`, y la de la sigmoide, `ŷ(1 − ŷ)`.

## 3. Gradiente de un peso de la capa de salida (`p₂₁`)

Se aplica la **regla de la cadena**, encadenando cómo el error depende de la salida, la salida de la suma, y la suma del peso:

```
∂E/∂p₂₁ = (∂E/∂ŷ) · (∂ŷ/∂sumaₛ) · (∂sumaₛ/∂p₂₁)
```

Resolviendo cada factor:

```
∂E/∂p₂₁ = (ŷ − y) · ŷ(1 − ŷ) · salida₀₁ = δₛ · salida₀₁
```

El producto de los dos primeros factores se agrupa y se le da un nombre: **`δₛ`** (delta de la neurona de salida). Ese agrupamiento es la clave de todo el algoritmo, porque se reutiliza:

```
∂E/∂p₂₂    = δₛ · salida₀₂
∂E/∂sesgo₂₁ = δₛ · 1
```

El sesgo se deriva igual que un peso cuya entrada vale siempre 1 — por eso en el diagrama los sesgos cuelgan de un nodo con el número 1.

## 4. Gradiente de un peso de la capa oculta (`p₁₁`)

Acá la cadena es más larga, porque `p₁₁` afecta al error a través de `O₁` y recién después a través de la salida:

```
∂E/∂p₁₁ = (∂E/∂ŷ) · (∂ŷ/∂sumaₛ) · (∂sumaₛ/∂salida₀₁) · (∂salida₀₁/∂suma₀₁) · (∂suma₀₁/∂p₁₁)
```

Cinco factores, que resueltos dan:

```
∂E/∂p₁₁ = (ŷ − y) · ŷ(1 − ŷ) · p₂₁ · salida₀₁(1 − salida₀₁) · x₁
```

Leída de izquierda a derecha, la fórmula es el recorrido del error hacia atrás: sale del error `(ŷ − y)`, atraviesa la activación de salida `ŷ(1 − ŷ)`, **cruza el peso `p₂₁` hacia la capa oculta**, atraviesa la activación de `O₁` —`salida₀₁(1 − salida₀₁)`, la derivada de la sigmoide— y termina en la entrada `x₁`.

Los dos primeros factores son otra vez `δₛ`: ya estaban calculados del paso anterior. **Eso es lo que hace eficiente al algoritmo**: no se deriva cada peso desde cero, se reutiliza lo ya calculado en las capas posteriores.

## 5. Los demás pesos y sesgos

Siguen el mismo patrón, cambiando qué neurona oculta y qué entrada intervienen:

```
∂E/∂p₁₁ = (ŷ − y) · ŷ(1 − ŷ) · p₂₁ · salida₀₁(1 − salida₀₁) · x₁
∂E/∂p₁₃ = (ŷ − y) · ŷ(1 − ŷ) · p₂₁ · salida₀₁(1 − salida₀₁) · x₂
∂E/∂p₁₂ = (ŷ − y) · ŷ(1 − ŷ) · p₂₂ · salida₀₂(1 − salida₀₂) · x₁
∂E/∂p₁₄ = (ŷ − y) · ŷ(1 − ŷ) · p₂₂ · salida₀₂(1 − salida₀₂) · x₂

∂E/∂sesgo₁₁ = (ŷ − y) · ŷ(1 − ŷ) · p₂₁ · salida₀₁(1 − salida₀₁) · 1
∂E/∂sesgo₁₂ = (ŷ − y) · ŷ(1 − ŷ) · p₂₂ · salida₀₂(1 − salida₀₂) · 1
```

Se ve el patrón: los pesos que llegan a `O₁` llevan `p₂₁` y `salida₀₁`; los que llegan a `O₂`, `p₂₂` y `salida₀₂`. El último factor es la entrada correspondiente (`x₁`, `x₂`, o `1` para los sesgos).

## 6. La actualización

Con los gradientes calculados, cada parámetro se actualiza con la regla del descenso de gradiente:

```
p₁₁^nuevo    = p₁₁^viejo    − η · (∂E/∂p₁₁)
sesgo₁₁^nuevo = sesgo₁₁^viejo − η · (∂E/∂sesgo₁₁)
```

donde `η` es la **tasa de aprendizaje**. Es exactamente la fórmula general de optimización, aplicada parámetro por parámetro.

## 7. Por qué importa el detalle de las derivadas

El factor `salida₀₁(1 − salida₀₁)` es la derivada de la sigmoide, y su valor **máximo es 0,25** (en `salida₀₁ = 0,5`); en los extremos tiende a 0. Como cada capa que el error atraviesa hacia atrás multiplica por un factor de ese tipo, en redes profundas el gradiente **se apaga**: es el problema del **gradiente desvaneciente**, y es la razón por la que ReLU reemplazó a la sigmoide en las capas ocultas.

## 8. Relación con el resto del módulo

- Es el algoritmo que **calcula el gradiente** que consume el descenso de gradiente.
- Generaliza a varias capas la **regla delta** del entrenamiento manual del perceptrón.
- La **saturación** de las funciones de activación se ve acá en su forma concreta: el factor de la derivada que multiplica en cada capa.
- En `scikit-learn` todo esto ocurre dentro de `fit()`; los gradientes no se exponen, pero `loss_curve_` muestra el resultado de aplicarlos época a época.
