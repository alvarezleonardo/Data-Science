# Funciones de pérdida

> Conversión a Markdown de las slides del curso (unidad "Optimización y regularización", bloque 01). El PDF original está en [`material/`](material/).

## 1. Qué es una función de pérdida

La **función de pérdida**, también conocida como **función de costo**, es una medida que cuantifica la **discrepancia entre los valores que el modelo predijo y los valores reales**.

Los puntos de la slide:

- Es **esencial** en el proceso de entrenamiento de una red neuronal.
- Produce **un único valor** que cuantifica el error (pérdida) de la predicción.
- Proporciona la **señal para ajustar los pesos y los sesgos** de la red.
- La elección de una función de pérdida adecuada **depende del tipo de problema**.
- Diferentes problemas pueden requerir **diferentes enfoques** para medir el error.

Ese valor único es lo que la retropropagación deriva para saber en qué dirección mover cada peso. Es la curva que `scikit-learn` guarda en `mlp.loss_curve_`.

## 2. Pérdidas para regresión

**Error Absoluto Medio (MAE)**

```
MAE = (1/n) · Σᵢ |yᵢ − ŷᵢ|
```

donde `yᵢ` es el valor real y `ŷᵢ` el valor predicho. Es el promedio de las distancias verticales entre cada punto y la recta ajustada.

**Error Cuadrático Medio (MSE)**

```
MSE = (1/n) · Σᵢ (yᵢ − ŷᵢ)²
```

Diferencia práctica entre las dos: el MSE **eleva al cuadrado**, así que castiga mucho más los errores grandes y es más sensible a los valores atípicos; el MAE es más **robusto** frente a outliers. El MSE, en cambio, es derivable en todo su dominio, lo que lo hace más cómodo para el descenso de gradiente. `MLPRegressor` minimiza MSE.

## 3. Pérdidas para clasificación

**Entropía binaria cruzada (BCE)** — clasificación **binaria**. Mide la diferencia entre dos distribuciones de probabilidad.

```
L(y, ŷ) = −( y·log(ŷ) + (1 − y)·log(1 − ŷ) )

L = −(1/N) · Σᵢ ( yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) )
```

**Entropía cruzada categórica (CCE)** — clasificación **multiclase**. Mide la diferencia entre la distribución verdadera y la distribución predicha por el modelo.

```
L(y, ŷ) = − Σᵢ₌₁..C  yᵢ·log(ŷᵢ)

L = −(1/N) · Σⱼ₌₁..N Σᵢ₌₁..C  yⱼᵢ·log(ŷⱼᵢ)
```

La primera fórmula de cada par es la pérdida de **una** muestra; la segunda es el promedio sobre las `N` muestras del lote. En la CCE, `C` es la cantidad de clases y `y` está en formato *one-hot*: como solo el término de la clase verdadera sobrevive, la pérdida es `−log` de la probabilidad que el modelo le asignó a la clase correcta.

## 4. Cómo elegir

| Problema | Pérdida | En scikit-learn |
|----------|---------|-----------------|
| Regresión | MSE (o MAE si hay outliers) | `MLPRegressor` (`loss='squared_error'`) |
| Clasificación binaria | BCE | `MLPClassifier` (log-loss) |
| Clasificación multiclase | CCE | `MLPClassifier` (log-loss) |

`MLPClassifier` no expone el parámetro: usa **log-loss** siempre, que es exactamente BCE en el caso binario y CCE en el multiclase.

**Pérdida no es lo mismo que métrica de evaluación.** La pérdida es lo que el modelo minimiza durante el entrenamiento y tiene que ser derivable; la métrica (exactitud, R², RMSE) es lo que se reporta. Coinciden a veces —MSE— y a veces no: nadie entrena minimizando exactitud, porque no es derivable.
