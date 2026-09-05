# Fundamentos de redes neuronales

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 8 | 28 | 🔄 En curso — teoría hasta el perceptrón multicapa + 3 notebooks propios y 3 recursos de clase |

> Módulo en curso. Documentado hasta la Clase 11 del programa (fundamentos biológicos, historia de las RNA, perceptrón, estructura y fórmulas, entrenamiento con la compuerta AND, limitaciones, implementación con scikit-learn y perceptrón multicapa). El resto del material (teoría, notebooks y datasets) se irá agregando a medida que avance.

## Teoría

Conversión a Markdown de las slides del curso, en [`teoria/`](teoria/):

- [`0 - Programa del módulo.md`](teoria/0%20-%20Programa%20del%20módulo.md)
- [`Fundamentos Biológicos.md`](teoria/Fundamentos%20Biológicos.md)
- [`Historia de las Redes Neuronales.md`](teoria/Historia%20de%20las%20Redes%20Neuronales.md)
- [`Perceptrón.md`](teoria/Perceptrón.md)
- [`Perceptrón - Estructura y Fórmulas.md`](teoria/Perceptrón%20-%20Estructura%20y%20Fórmulas.md)
- [`Compuerta Lógica AND - Parte 1.md`](<teoria/Compuerta Lógica AND - Parte 1.md>) a [`Parte 4.md`](<teoria/Compuerta Lógica AND - Parte 4.md>)
- [`Limitaciones.md`](teoria/Limitaciones.md)
- [`Implementación con Scikit-Learn.md`](<teoria/Implementación con Scikit-Learn.md>)
- [`Perceptrón Multicapa.md`](<teoria/Perceptrón Multicapa.md>)

## Notebooks

- [`practica_redes_neuronales.ipynb`](<notebooks/practica_redes_neuronales.ipynb>) — implementación del perceptrón desde cero: función escalón, regla delta, entrenamiento con la compuerta AND (reproduciendo los pesos del cálculo manual de las slides), frontera de decisión, el problema del XOR y la versión con `sklearn.linear_model.Perceptron`.
- [`perceptron_iris_sklearn.ipynb`](<notebooks/perceptron_iris_sklearn.ipynb>) — el perceptrón aplicado al dataset **Iris** (setosa vs versicolor) con `scikit-learn`: exploración del dataset, división train/test estratificada, entrenamiento, lectura de pesos y sesgo, exactitud, matriz de confusión, frontera de decisión, y el contraste entre la compuerta AND (converge) y el XOR (no puede). Amplía el recurso descargable de la clase `OD_RN1_ESP_M02_S05` con explicaciones paso a paso y corrige el cálculo de exactitud del original.
- [`mlp_clasificacion_iris.ipynb`](<notebooks/mlp_clasificacion_iris.ipynb>) — **perceptrón multicapa** aplicado a las 3 clases de Iris con `MLPClassifier`: escalado, diagnóstico del entrenamiento (`n_iter_`, `loss_curve_`), inspección de la arquitectura real, regiones de decisión no lineales, efecto del tamaño de la red y comparación directa contra el perceptrón simple. Amplía el recurso de clase `OD_RN1_ESP_M03_S10`.
- [`mlp_regresion_california.ipynb`](<notebooks/mlp_regresion_california.ipynb>) — **MLP para regresión** sobre California Housing con `MLPRegressor`: exploración del dataset, por qué el escalado es obligatorio, MSE/RMSE/MAE/R² y cómo leerlas, comparación contra la media y una regresión lineal, análisis de residuos y del techo artificial del dataset. Amplía el recurso de clase `OD_RN1_ESP_M03_S11`.

Los recursos descargables originales de la clase se conservan sin modificar en [`notebooks/`](notebooks/), con su nombre `OD_RN1_ESP_*`.

Apuntes consolidados también en el [documento maestro](../APUNTES-DATA-SCIENCE.md#módulo-06--fundamentos-de-redes-neuronales), secciones 25 a 31.

[← Volver al índice](../README.md)
