# Perceptrón: Estructura y Fórmulas

> Conversión a Markdown de la slide del curso. El PDF original está en [`material/`](material/).

## 1. Estructura del perceptrón

El perceptrón se compone de los siguientes elementos:

- **Entradas** (`x1, x2, x3, ..., xn`): los valores de las características de la observación.
- **Pesos** (`w1, w2, w3, ..., wn`): un peso asociado a cada entrada, que determina su importancia relativa.
- **Umbral** (`θ`, theta): un término independiente (bias) que se suma junto con las entradas ponderadas.
- **Suma ponderada** (`Σ`): combina las entradas multiplicadas por sus pesos, más el umbral.
- **Función de activación** (`Φ`): transforma la suma ponderada en la salida final del perceptrón.
- **Salida** (`y`): el resultado de la clasificación (binaria).

Esquema del flujo: `x1, x2, ..., xn` (con sus pesos `w1, w2, ..., wn`) y `θ` → suma ponderada `Σ` → función de activación `Φ` → salida `y`.

## 2. Fórmula de la suma ponderada

La suma ponderada `z` combina cada entrada con su peso correspondiente y le suma el umbral:

```
z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + θ
```

De forma equivalente, usando notación de sumatoria:

```
z = Σ (i=1 a n) wi*xi + θ
```

## 3. Función de activación

El perceptrón utiliza una **función escalón** (step function) como función de activación `Φ`: la salida es 1 si la suma ponderada supera cero, y 0 en caso contrario.

```
Φ(z) = 1  si z > 0
Φ(z) = 0  en caso contrario
```

La salida del perceptrón (`y`) es directamente el resultado de aplicar esta función de activación a `z`:

```
y = Φ(z) = 1  si z > 0
y = Φ(z) = 0  en caso contrario
```

## 4. Actualización de pesos (aprendizaje)

El perceptrón aprende ajustando iterativamente sus pesos según el error entre la salida predicha (`y`) y la salida real (`y_real`):

```
w_i(nuevo) = w_i(anterior) + Δw_i
Δw_i = n * (y_real - y) * x_i
```

Donde `n` (eta) es la **tasa de aprendizaje**, que regula cuánto se ajusta el peso en cada actualización.

## 5. Actualización del umbral

De manera análoga a los pesos, el umbral `θ` también se actualiza en función del error de predicción:

```
θ(nuevo) = θ(anterior) + Δθ
Δθ = n * (y_real - y)
```

> **Nota:** la slide presenta la actualización de `w_i` y de `θ` como fórmulas separadas y sucesivas (regla de aprendizaje del perceptrón). Ambas se aplican en cada iteración de entrenamiento, cada vez que la predicción `y` difiere del valor real `y_real`.
