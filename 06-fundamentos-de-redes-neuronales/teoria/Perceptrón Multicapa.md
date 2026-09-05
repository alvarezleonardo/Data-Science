# Perceptrón Multicapa

> Conversión a Markdown de la slide del curso. El PDF original está al lado.

> **Nota:** el PDF de origen es un deck con animaciones tipo "build": la slide "Perceptrón Multicapa" se repite agregando un ítem por página. Se consolidó en una sola sección con los cuatro puntos, evitando duplicar el texto repetido. La slide "Arquitectura" es un diagrama; se describe en texto más abajo.

## 1. Qué es un perceptrón multicapa

Un **perceptrón multicapa (MLP)** es un tipo de red neuronal artificial que se utiliza para abordar problemas complejos de **clasificación** y **regresión**.

A diferencia del perceptrón simple, que solo puede resolver problemas **linealmente separables**, el MLP puede manejar **relaciones no lineales** entre las características de entrada y las salidas deseadas.

Los cuatro puntos que plantea la slide:

- El proceso de entrenamiento de un MLP generalmente se realiza mediante el algoritmo de **retropropagación**.
- Se destaca su capacidad para **aprender y modelar relaciones complejas**, gracias a su estructura de capas y a las **funciones de activación no lineales**.
- Las redes neuronales multicapa son **sensibles a la calidad y la naturaleza de los datos** de entrada.
- Necesitan una **gran cantidad de datos** para entrenar eficazmente, y tienen **riesgo de sobreajuste**.

## 2. Arquitectura

La red se organiza en capas de neuronas conectadas entre sí:

- **Capa de entrada:** una neurona por cada característica del dataset. No hace cálculo, solo recibe los datos.
- **Capas ocultas:** una o más. Son las que le dan al modelo la capacidad de representar relaciones no lineales. Cada neurona calcula `z = Σ wᵢ·xᵢ + b` y le aplica una función de activación no lineal.
- **Capa de salida:** una neurona en regresión (valor continuo); en clasificación, una por clase.

Sobre esa arquitectura corren dos procesos complementarios:

- **Forward Propagation (propagación hacia delante):** es el proceso mediante el cual los datos de entrada se transforman a través de las capas de la red hasta producir una **salida final**.
- **Backpropagation (retropropagación):** es el proceso de **ajuste de los pesos y sesgos** de la red para **minimizar la función de pérdida**.

## 3. Por qué las capas ocultas necesitan activación no lineal

Es el punto que justifica toda la arquitectura. Si las neuronas de las capas ocultas solo hicieran la suma ponderada, sin función de activación no lineal, **la composición de varias capas lineales sigue siendo una función lineal**: la red entera colapsaría al equivalente de un perceptrón simple, con la misma limitación (§29).

La no linealidad (ReLU, tanh, logística) es lo que permite que apilar capas agregue capacidad de representación real, y con eso resolver problemas como el **XOR**, que el perceptrón simple no puede.

## 4. Relación con el resto del módulo

- El perceptrón simple (§27) es el bloque base: una sola neurona, frontera de decisión lineal.
- Sus **limitaciones** (§29) —solo problemas linealmente separables, el caso del XOR— son exactamente lo que el MLP viene a resolver.
- La **regla delta** del entrenamiento manual (§28) es el caso más simple de lo que la retropropagación generaliza a varias capas.
- En `scikit-learn` la implementación son los estimadores `MLPClassifier` y `MLPRegressor`, con la misma API `fit` / `predict` (§30). Para redes más profundas se pasa a TensorFlow o PyTorch.

## 5. Contrapartida: qué se paga

El MLP no es gratis respecto del perceptrón simple:

| | Perceptrón simple | MLP |
|---|---|---|
| Problemas que resuelve | solo linealmente separables | también no lineales |
| Interpretabilidad | alta: 2 pesos y un sesgo legibles | baja: cientos o miles de pesos sin lectura directa |
| Datos necesarios | pocos | muchos |
| Riesgo de sobreajuste | bajo (modelo muy rígido) | alto: requiere regularización (`alpha`) y validación |
| Sensibilidad a la escala | tolerable | alta: **hay que estandarizar** las entradas |
| Costo de entrenamiento | trivial | significativo |
