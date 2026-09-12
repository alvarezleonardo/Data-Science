# Introducción a PyTorch - Parte 3

> Conversión a Markdown de las slides del curso (Clase 4 — Introducción a PyTorch, bloque 3). El PDF original está en [`material/`](material/).

## 1. Definir un modelo

Para crear un modelo de red neuronal, hay que definirlo a través de una clase que herede de **`torch.nn.Module`**. Es la clase base en PyTorch para todos los modelos de redes neuronales, y provee una estructura estándar para construir, entrenar y evaluar redes neuronales.

```python
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # Neural Network Building
    def forward(self, x):
        # Forward operations
        return x
```

Toda subclase de `nn.Module` define dos métodos:

- **`__init__`**: acá se declaran las **capas** que va a tener la red (como atributos del objeto). Siempre empieza llamando a `super().__init__()` para inicializar correctamente la clase base.
- **`forward`**: acá se define **cómo fluyen los datos** por esas capas, es decir, el cálculo real que hace la red al recibir una entrada `x`.

## 2. Un MLP concreto

La slide arma una pequeña red neuronal (perceptrón multicapa, MLP) con dos capas totalmente conectadas:

```python
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(8, 10)
        self.layer2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```

`nn.Linear(8, 10)` es una capa densa que recibe 8 entradas y produce 10 salidas; `nn.Linear(10, 3)` toma esas 10 salidas y produce 3, típicamente la cantidad de clases del problema (por ejemplo, las 3 clases de Iris en el notebook del curso).

La slide siguiente agrega funciones de activación entre las capas:

```python
def forward(self, x):
    x = self.layer1(x)
    x = torch.relu(x)
    x = self.layer2(x)
    x = torch.softmax(x, dim=1)
    return x
```

> **Nota:** en la slide original, `torch.relu()` y `torch.softmax()` aparecen sin argumentos (`torch.relu()`, `torch.softmax()`), lo cual no es código válido: ambas funciones necesitan recibir el tensor sobre el que operan, y `softmax` además necesita la dimensión (`dim`) sobre la que normaliza. Arriba se corrigió agregando `x` y `dim=1`.

> **Nota — por qué NO conviene este `softmax` final:** si el modelo se va a entrenar con `nn.CrossEntropyLoss` (como en la sección 2), la capa de salida **no debe** aplicar `softmax`. `CrossEntropyLoss` combina internamente `log_softmax` + `NLLLoss`, así que si el modelo ya devuelve probabilidades vía `softmax`, se le estaría aplicando softmax dos veces: una en el modelo y otra dentro de la loss. Esto no tira error, pero distorsiona el gradiente y el entrenamiento aprende mal o más lento. Es uno de los errores más comunes al empezar con PyTorch. La regla práctica: con `CrossEntropyLoss`, la última capa del modelo devuelve los **logits crudos** (sin activación), tal como en el primer bloque de código de esta sección.

## 3. Función de pérdida

Hay que seleccionar la función de pérdida o error (*loss*). En PyTorch están definidas en **`torch.nn`**:

| Clase | Para qué se usa |
|---|---|
| `torch.nn.BCELoss` (*binary cross entropy*) | clasificación **binaria** |
| `torch.nn.CrossEntropyLoss` (*cross entropy*) | clasificación **multiclase** |
| `torch.nn.MSELoss` (*mean squared error*) | problemas de **regresión** |

```python
criterion = nn.CrossEntropyLoss()
```

## 4. Algoritmo de optimización

Hay que seleccionar y parametrizar el algoritmo de optimización. En PyTorch los optimizadores están definidos en **`torch.optim`**:

- `torch.optim.SGD()` implementa el **Descenso de Gradiente Estocástico**.
- `torch.optim.Adam()` implementa **Adam**.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`model.parameters()` le pasa al optimizador todos los pesos y sesgos del modelo que tiene que actualizar, y `lr` es la tasa de aprendizaje.

`SGD` incorpora el parámetro **`momentum`**. Esto ayuda a acelerar el aprendizaje y a evitar quedarse atrapado en mínimos locales, al tener en cuenta el gradiente de los pasos anteriores.

## 5. Relación con el resto del módulo

- El modelo (`nn.Module`), la `criterion` (loss) y el `optimizer` definidos acá son las tres piezas que se usan en el bucle de entrenamiento de la Parte 4 (`loss = criterion(...)`, `optimizer.step()`).
- Los `DataLoader` construidos en la Parte 2 son los que alimentan el `forward` de este modelo con lotes de datos.
- El parámetro `momentum` de `SGD` y la elección de la tasa de aprendizaje se explican en detalle, para el caso general de redes neuronales, en [`06-fundamentos-de-redes-neuronales/teoria/Optimización.md`](../../06-fundamentos-de-redes-neuronales/teoria/Optimizaci%C3%B3n.md).
- El notebook [`OD_RN2_ESP_M02_S05_Implementación en PyTorch`](../notebooks/OD_RN2_ESP_M02_S05_Implementación%20en%20PyTorch%20-%20Recurso%20descargable.ipynb) define un modelo, una `CrossEntropyLoss` y un optimizador siguiendo exactamente este patrón sobre el dataset Iris.
