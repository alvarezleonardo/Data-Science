# Implementación con Scikit-Learn

> Conversión a Markdown de la slide del curso. El PDF original está al lado.

> **Nota:** el PDF de origen es un deck con animaciones tipo "build": la slide "Introducción a Scikit-Learn" se repite 4 veces, agregando un párrafo por página. Se consolidó en una sola sección con los tres puntos, evitando duplicar el texto repetido.

## 1. Introducción a Scikit-Learn

Scikit-Learn es una de las librerías más populares y ampliamente utilizadas en el ecosistema de Python para el **aprendizaje automático**. Las ideas que plantea la slide son tres:

- **Popularidad y adopción:** es una de las librerías de referencia del ecosistema Python para machine learning.
- **Amplia gama de algoritmos y utilidades:** sus funcionalidades permiten abordar **todo el flujo de trabajo** de aprendizaje automático, desde la preparación de los datos hasta la evaluación del modelo.
- **Integración con el ecosistema:** se combina fácilmente con otras bibliotecas de Python, incluidas las de aprendizaje profundo como **TensorFlow** y **PyTorch**.

## 2. Qué implica para el perceptrón

Aplicado al contenido de este módulo: el entrenamiento que en las partes 1 a 4 de la compuerta AND se hizo **a mano** (calcular `z`, aplicar la función escalón, ajustar pesos y umbral iteración por iteración) queda encapsulado en el estimador `sklearn.linear_model.Perceptron`, que implementa la misma regla de aprendizaje.

```python
from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # x1, x2
y = np.array([0, 0, 0, 1])                        # x1 AND x2

clf = Perceptron(max_iter=1000, eta0=0.045, random_state=42)
clf.fit(X, y)

clf.predict(X)              # array([0, 0, 0, 1]) → aprende la compuerta AND
clf.coef_, clf.intercept_   # pesos (w1, w2) y bias
```

El flujo `fit` / `predict` es el mismo para cualquier estimador de la librería, que es justamente el punto de la slide: una API uniforme que cubre el ciclo completo del modelo.

## 3. Relación con el resto del módulo

- La implementación manual sirve para **entender la mecánica** (regla delta, convergencia, rol de la tasa de aprendizaje).
- Scikit-Learn sirve para **trabajar en la práctica**, con validación, métricas y preprocesamiento integrados.
- Para redes más profundas (múltiples capas, backpropagation) el ecosistema se apoya en TensorFlow o PyTorch, con los que Scikit-Learn se integra.
