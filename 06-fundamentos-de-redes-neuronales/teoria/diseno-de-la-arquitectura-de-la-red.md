# Diseño de la arquitectura de la red

> Conversión a Markdown de las slides del curso (unidad "Funciones de activación y estructura de la red", bloque 03). El PDF original está en [`material/`](material/).

## 1. Por qué importa

Seleccionar adecuadamente el **número de capas** y la **cantidad de neuronas** es muy importante para el rendimiento y la eficiencia del modelo. La arquitectura influye directamente en tres cosas:

- su **capacidad de aprendizaje**,
- su **poder de generalización**,
- su **rendimiento computacional**.

## 2. Número de capas

- Aumentar la cantidad de capas permite a la red aprender **representaciones más complejas** de los datos.
- Mayor cantidad de capas aumenta la **complejidad de la red** y el **tiempo de procesamiento** del entrenamiento.

## 3. Número de neuronas

- **Demasiadas neuronas** pueden llevar a **sobreajuste** (*overfitting*): la red memoriza el conjunto de entrenamiento y generaliza mal.
- **Pocas neuronas** dejan a la red sin capacidad suficiente para captar la complejidad de los datos (*underfitting*).

## 4. Cómo se traduce en la práctica

No hay fórmula: la arquitectura es un **hiperparámetro** y se elige comparando en validación, no en train.

- Arrancar simple (una capa oculta) y agrandar solo si el error de entrenamiento sigue alto.
- Si el error de train es bajo pero el de test es alto, el problema es sobreajuste: achicar la red, o regularizar (`alpha` en `scikit-learn`) antes que agregar capas.
- Recordar que la capa densa (§33) hace crecer los parámetros como el producto de los tamaños: pasar de `(50,)` a `(100, 100)` no duplica el costo, lo multiplica.
- En `scikit-learn` la arquitectura se declara con `hidden_layer_sizes`, y conviene verificar lo que realmente quedó entrenado con `mlp.n_layers_`, `mlp.coefs_` y la curva `mlp.loss_curve_`.
