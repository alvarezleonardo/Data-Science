# Introducción a PyTorch - Parte 2

> Conversión a Markdown de las slides del curso (Clase 4 — Introducción a PyTorch, bloque 2). El PDF original está en [`material/`](material/).

## 1. Acceso a los datos

PyTorch ofrece una serie de clases para la gestión y el acceso a los datos. **`torchvision`** es una biblioteca complementaria a PyTorch que está específicamente diseñada para tareas de **visión por computadora**: trae datasets ya empaquetados y transformaciones listas para usar.

```python
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True,
                                         transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True,
                                        transform=transforms.ToTensor())
```

`CIFAR10` es uno de los datasets de referencia que trae `torchvision.datasets`: 60.000 imágenes en color de 10 clases. `root` indica dónde se descarga/busca el dataset, `train` distingue el split de entrenamiento del de test, y `transform` aplica una transformación a cada imagen al leerla.

## 2. Transformaciones disponibles

`torchvision` incluye un conjunto de transformaciones comunes para preprocesar los datos y adaptarlos a los problemas o redes neuronales que se quieran emplear:

| Transformación | Qué hace |
|---|---|
| `ToTensor` | Transforma los datos/imágenes en tensores. |
| `Resize` | Cambia el tamaño de la imagen a un tamaño dado. |
| `Normalize` | Normaliza los valores dada una media y desviación estándar. |
| `CenterCrop` | Corta la imagen dado un tamaño desde el centro. |
| `RandomHorizontalFlip` | Gira horizontalmente una imagen de forma aleatoria. |
| `RandomRotation` | Rota una imagen unos grados dados de forma aleatoria. |
| `Grayscale` | Transforma una imagen a escala de grises. |
| `Compose` | Aplica una serie de transformaciones especificadas en una lista. |

## 3. Encadenar transformaciones con `Compose`

`transforms.Compose` permite aplicar varias transformaciones en secuencia, como un pipeline:

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./train', train=True,
                                         download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./test', train=False,
                                        download=True, transform=transform)
```

Acá primero se convierte la imagen a tensor y después se normaliza con media `0.5` y desvío estándar `0.5` en cada uno de los tres canales (RGB), llevando los valores del rango `[0, 1]` al rango `[-1, 1]`.

## 4. `DataLoader`: leer los datos por lotes

Para entrenar hace falta definir un **iterador** que permita la lectura de los conjuntos de datos por lotes (*batches*). Para eso se crea un objeto de la clase `DataLoader`.

```python
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)
```

| Parámetro | Qué controla |
|---|---|
| `batch_size` | cuántas muestras entrega el `DataLoader` en cada iteración (acá, 8) |
| `shuffle` | si mezcla el orden de las muestras en cada época |

`shuffle=True` en el set de entrenamiento evita que la red vea siempre los datos en el mismo orden (lo que podría sesgar el aprendizaje); en el set de test no hace falta mezclar, porque solo se usa para evaluar.

## 5. Relación con el resto del módulo

- El `trainloader` y el `testloader` que se arman acá son exactamente lo que se itera en el bucle de entrenamiento (`for input, target in trainloader:`) y en la evaluación de la Parte 4.
- La Parte 3 define el modelo (`nn.Module`) que va a recibir esos lotes de datos.
- En el notebook de referencia del curso, [`OD_RN2_ESP_M02_S05_Implementación en PyTorch`](../notebooks/OD_RN2_ESP_M02_S05_Implementación%20en%20PyTorch%20-%20Recurso%20descargable.ipynb), el dataset es Iris (tabular) en vez de CIFAR10 (imágenes), pero el patrón `Dataset` + `DataLoader` es el mismo.
