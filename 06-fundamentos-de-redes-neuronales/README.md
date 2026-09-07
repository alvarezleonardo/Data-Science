# Fundamentos de redes neuronales

| Unidades | Clases | Estado |
|:--------:|:------:|--------|
| 8 | 28 | 🔄 En curso — teoría hasta backpropagation y persistencia de modelos + 4 notebooks propios y 7 recursos de clase |

> Módulo en curso. Documentado hasta la unidad de backpropagation y gestión de modelos (fundamentos biológicos, historia de las RNA, perceptrón, estructura y fórmulas, entrenamiento con la compuerta AND, limitaciones, implementación con scikit-learn, perceptrón multicapa, funciones de activación, grafos y capa densa, diseño de la arquitectura, funciones de pérdida, optimización, regularización, backpropagation y persistencia de modelos). El resto del material (teoría, notebooks y datasets) se irá agregando a medida que avance.

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
- [`Introducción a las funciones de activación.md`](<teoria/Introducción a las funciones de activación.md>)
- [`Grafos y capa densa.md`](<teoria/Grafos y capa densa.md>)
- [`Diseño de la arquitectura de la red.md`](<teoria/Diseño de la arquitectura de la red.md>)
- [`Funciones de Pérdida.md`](<teoria/Funciones de Pérdida.md>)
- [`Optimización.md`](<teoria/Optimización.md>)
- [`Regularización.md`](<teoria/Regularización.md>)
- [`Backpropagation.md`](<teoria/Backpropagation.md>)
- [`Gestión de modelos.md`](<teoria/Gestión de modelos.md>)

## Notebooks

- [`practica_redes_neuronales.ipynb`](<notebooks/practica_redes_neuronales.ipynb>) — implementación del perceptrón desde cero: función escalón, regla delta, entrenamiento con la compuerta AND (reproduciendo los pesos del cálculo manual de las slides), frontera de decisión, el problema del XOR y la versión con `sklearn.linear_model.Perceptron`.
- [`perceptron_iris_sklearn.ipynb`](<notebooks/perceptron_iris_sklearn.ipynb>) — el perceptrón aplicado al dataset **Iris** (setosa vs versicolor) con `scikit-learn`: exploración del dataset, división train/test estratificada, entrenamiento, lectura de pesos y sesgo, exactitud, matriz de confusión, frontera de decisión, y el contraste entre la compuerta AND (converge) y el XOR (no puede). Amplía el recurso descargable de la clase `OD_RN1_ESP_M02_S05` con explicaciones paso a paso y corrige el cálculo de exactitud del original.
- [`mlp_clasificacion_iris.ipynb`](<notebooks/mlp_clasificacion_iris.ipynb>) — **perceptrón multicapa** aplicado a las 3 clases de Iris con `MLPClassifier`: escalado, diagnóstico del entrenamiento (`n_iter_`, `loss_curve_`), inspección de la arquitectura real, regiones de decisión no lineales, efecto del tamaño de la red y comparación directa contra el perceptrón simple. Amplía el recurso de clase `OD_RN1_ESP_M03_S10`.
- [`mlp_regresion_california.ipynb`](<notebooks/mlp_regresion_california.ipynb>) — **MLP para regresión** sobre California Housing con `MLPRegressor`: exploración del dataset, por qué el escalado es obligatorio, MSE/RMSE/MAE/R² y cómo leerlas, comparación contra la media y una regresión lineal, análisis de residuos y del techo artificial del dataset. Amplía el recurso de clase `OD_RN1_ESP_M03_S11`.

- [`funciones_activacion.ipynb`](<notebooks/funciones_activacion.ipynb>) — **funciones de activación** en detalle: sigmoide, tanh y ReLU graficadas desde cero con sus derivadas, la saturación medida numéricamente (derivada ~0,0025 y ~0,00002 en `x = 6`, contra 1 de ReLU), la demostración empírica de que con `activation='identity'` un MLP no resuelve el XOR y con `tanh`/`relu` sí, la comparación de las tres activaciones sobre Iris **con y sin escalado** contra el recurso de clase, y el efecto de la arquitectura (de `(2,)` a `(100,100)`) sobre capacidad, sobreajuste y cantidad de parámetros. Amplía el recurso de clase `OD_RN1_ESP_M03_S10`.

Los recursos descargables de la clase se conservan en [`notebooks/`](notebooks/) con su nombre `OD_RN1_ESP_*`. Dos aclaraciones sobre ellos:

- **`OD_RN1_ESP_M03_S10` acumula tres clases**: el `MLPClassifier` básico, la comparación de funciones de activación y el bloque de optimización y regularización (curva de pérdida, barrido de `alpha`, solvers y `GridSearchCV`). Se le agregaron celdas markdown explicando cada bloque; el código y los outputs originales quedaron intactos. Sus outputs de las celdas 14 a 18 corresponden a una red `(50,50,50)` con `tanh` y no al código `(100,200,300)` con `relu` que figura, así que reejecutarlo da resultados distintos.
- **`OD_RN1_ESP_M04_S15`** se descarga con extensión `.json`, pero es un notebook `nbformat`; se guarda como `.ipynb`.

Apuntes consolidados también en el [manual del programa](../APUNTES-DATA-SCIENCE.md#parte-vii--redes-neuronales), capítulos 24 a 38.

### Persistencia de modelos

Los notebooks `guardar_modelo_joblib_pickle.ipynb`, `cargar_modelo_joblib.ipynb` y `cargar_modelo_pickle.ipynb` son los recursos de las clases 23-24. Se les agregaron celdas markdown explicativas, sin tocar el código ni los outputs.

> Los archivos de modelo que generan (`.joblib`, `.pkl`) **no se versionan** — están en el `.gitignore` como todo artefacto binario. Para correr los dos notebooks de carga hay que ejecutar antes `guardar_modelo_joblib_pickle.ipynb`.

## Práctica

- [`practica/evaluacion-final/`](<practica/evaluacion-final/>) — trabajo de la **evaluación final** sobre el dataset `winemag-data-130k-v2.csv`: enunciado y notebook de resolución. El dataset no se versiona; ver el README de esa carpeta.

[← Volver al índice](../README.md)
