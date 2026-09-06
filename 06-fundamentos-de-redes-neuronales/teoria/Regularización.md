# Regularización

> Conversión a Markdown de las slides del curso (unidad "Optimización y regularización", bloque 02). El PDF original está al lado.

> **Nota sobre el PDF:** el archivo tiene 75 páginas, pero el contenido de la clase son solo las **8 primeras**. De la página 9 en adelante quedó adjunta por error la **plantilla de slides institucional** de Digital House (texto *lorem ipsum*, ejemplos de SQL y JavaScript, guías de estilo, assets). Se ignora todo eso. El deck también repite la misma slide agregando un ítem por página; se consolidó.

## 1. Qué es la regularización

La **regularización** es un conjunto de técnicas utilizadas para **prevenir el sobreajuste (overfitting)** y **mejorar la capacidad de generalización** del modelo.

Los tres puntos de la slide:

- Al aplicar técnicas de regularización, se **controla la complejidad del modelo** y se evita que los pesos y parámetros se vuelvan **demasiado grandes o especializados** en los datos de entrenamiento.
- **No existe una técnica de regularización universalmente superior.**
- La implementación de técnicas de regularización **puede aumentar el tiempo de entrenamiento** de la red.

## 2. Las tres técnicas

| Técnica | Qué hace | Fórmula |
|---|---|---|
| **L1** | agrega un término de penalización **proporcional a la suma de los valores absolutos** de los pesos | `L₁(X, w) = L(X, w) + λ · Σ \|wᵢ\|` |
| **L2** | el término de penalización es **proporcional a la suma de los valores al cuadrado** de los pesos | `L₂(X, w) = L(X, w) + λ · Σ wᵢ²` |
| **Dropout** | **apaga aleatoriamente un porcentaje de neuronas** durante el entrenamiento | — |

En L1 y L2, `L(X, w)` es la pérdida original (§35) y `λ` gradúa cuánto pesa la penalización: con `λ = 0` no hay regularización; cuanto más grande, más se fuerza a que los pesos sean chicos.

**La diferencia entre L1 y L2** está en qué le hace cada una a los pesos: L1 tiende a llevar pesos directamente a **cero** (selecciona variables, produce redes ralas), mientras que L2 los **encoge** hacia cero sin anularlos. Es la misma distinción que Lasso y Ridge en regresión lineal (§21).

**Dropout** es distinto en naturaleza: no toca la función de pérdida, sino la arquitectura durante el entrenamiento. Al apagar neuronas al azar en cada paso, impide que la red dependa demasiado de una neurona en particular y la obliga a distribuir la representación. La slide lo dibuja como dos redes: una completa y otra con varias neuronas marcadas en rojo (apagadas).

## 3. Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Ayuda a prevenir el **sobreajuste** | Requiere **mayor poder computacional** durante el entrenamiento |
| Permite **controlar la complejidad** del modelo | Suma **hiperparámetros** que hay que elegir |
| Puede **acelerar la convergencia** durante el entrenamiento | Puede producirse una **pérdida parcial de información** |

## 4. En scikit-learn

`MLPClassifier` y `MLPRegressor` implementan **solo L2**, mediante el parámetro **`alpha`** (que es la `λ` de la fórmula). Su valor por defecto es `0.0001`.

```python
MLPClassifier(hidden_layer_sizes=(100,), alpha=0.01)   # más regularizado
```

**No hay L1 ni dropout** en `scikit-learn` para redes: para eso hay que pasar a TensorFlow/Keras o PyTorch, donde dropout es una capa más (`Dropout(0.5)`).

Hay además una forma de regularización que no aparece en la slide y que sí está disponible: la **parada temprana** (`early_stopping=True`, junto con `n_iter_no_change` y `validation_fraction`). Cortar el entrenamiento cuando la métrica de validación deja de mejorar evita que la red siga ajustándose al ruido del train.

> **Cómo se busca `alpha` en la práctica:** con validación cruzada, graficando exactitud de train y de validación contra `alpha`. La señal de sobreajuste es la brecha entre las dos curvas; el `alpha` bueno es el que la cierra sin hundir las dos. Un barrido de `alpha` sobre un problema demasiado fácil no muestra nada: en el recurso de clase `OD_RN1_ESP_M03_S10`, cinco valores que cubren cinco órdenes de magnitud dan todos la misma exactitud, porque Iris con 2 atributos no llega a sobreajustar.
