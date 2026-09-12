# Introducción a las funciones de activación

> Conversión a Markdown de las slides del curso (unidad "Funciones de activación y estructura de la red", bloque 01). El PDF original está en [`material/`](material/).

> **Nota:** las slides de cada función son imágenes (curva + fórmula) sin texto. Se transcriben las fórmulas y se describe la forma de cada curva.

## 1. Qué son y para qué sirven

Las **funciones de activación** determinan la salida de una neurona. Sin función de activación, una red neuronal sería simplemente una **combinación lineal de las entradas**, y apilar capas no agregaría capacidad (§31).

Los cuatro puntos de la slide:

- Introducen **no linealidad** en la red.
- Determinan si una neurona **debe activarse** (pasar su información a la siguiente capa) o no.
- Algunas **normalizan las salidas a un rango específico**.
- Afectan **la forma en que se actualizan los pesos** (porque la retropropagación usa su derivada).

## 2. Sigmoide (logística)

```
f(x) = 1 / (1 + e^(-x))
```

- **Rango:** (0, 1). Curva en forma de S; vale 0,5 en `x = 0`, se aplana hacia 0 por izquierda y hacia 1 por derecha.
- **Uso típico:** capa de salida en clasificación binaria, donde la salida se lee como probabilidad.
- **Problema:** en los extremos la curva es casi plana, así que la derivada tiende a 0 (**saturación**). En redes profundas eso hace que el gradiente se desvanezca y las capas iniciales casi no aprendan.

> **Errata de la slide:** el PDF muestra `f(x) = 1 / (1 − e^(−x))`, con signo menos en el denominador. Es un error tipográfico: la sigmoide lleva **más** (`1 + e^(−x)`). Con el signo menos la función ni siquiera está acotada en (0, 1) —se indefine en `x = 0`— y no coincide con la curva dibujada en la misma slide. En `scikit-learn` esta activación es `activation='logistic'`.

## 3. Tangente hiperbólica (tanh)

```
φ(z) = tanh(z) = sinh(z) / cosh(z) = (e^z − e^(−z)) / (e^z + e^(−z))
```

- **Rango:** (−1, 1). Misma forma de S que la sigmoide, pero **centrada en 0**.
- **Ventaja sobre la sigmoide:** al estar centrada en cero, las salidas de una capa no arrastran un sesgo positivo hacia la siguiente, y la convergencia suele ser más rápida.
- **Comparte el problema de la saturación** en los extremos.

## 4. Unidad lineal rectificada (ReLU)

```
f(x) = 0   si x < 0
f(x) = x   si x ≥ 0
```

- **Rango:** [0, ∞). Es una recta quebrada: plana en cero para las entradas negativas, identidad para las positivas.
- **Ventaja:** para `x > 0` la derivada es constante (1), así que **no satura** por el lado positivo. Es barata de calcular y es la opción por defecto en capas ocultas.
- **Problema:** las neuronas cuya entrada queda siempre negativa tienen gradiente 0 y dejan de aprender (*dying ReLU*).

## 5. Cuál usar

| Capa | Elección habitual |
|------|-------------------|
| Ocultas | **ReLU** por defecto; `tanh` si la red es chica o los datos están centrados |
| Salida — regresión | Ninguna (identidad) |
| Salida — clasificación binaria | **Sigmoide** |
| Salida — clasificación multiclase | **Softmax** |

En `scikit-learn`, el parámetro `activation` de `MLPClassifier` / `MLPRegressor` acepta `'relu'` (default), `'tanh'`, `'logistic'` e `'identity'`, y **aplica solo a las capas ocultas**: la activación de salida la elige el estimador según el problema.
