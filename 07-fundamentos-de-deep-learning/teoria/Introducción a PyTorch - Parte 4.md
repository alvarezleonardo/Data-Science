# Introducción a PyTorch - Parte 4

> Conversión a Markdown de las slides del curso (Clase 4 — Introducción a PyTorch, bloque 4). El PDF original está al lado.

> **Nota:** el deck original repite el mismo bloque de código en 9 páginas seguidas, cada una señalando con una anotación una línea distinta (efecto "build" de una presentación). Se consolida acá en una sola sección con el código completo y una tabla línea por línea.

## 1. El bucle de entrenamiento completo

```python
model = MyNeuralNetwork()
for epoch in range(2):
    for input, target in trainloader:
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
```

## 2. Qué hace cada línea

| Línea | Qué hace | Si se la olvida |
|---|---|---|
| `model = MyNeuralNetwork()` | Crea una **instancia** del modelo definido en la Parte 3. | No hay modelo que entrenar; error antes de llegar al bucle. |
| `for epoch in range(2):` | Itera por un número de **épocas** de entrenamiento (acá, 2): cada época recorre una vez todo el dataset de entrenamiento. | Con menos épocas de las necesarias, el modelo queda subentrenado (no convergió). |
| `for input, target in trainloader:` | Se realiza para **cada batch** del conjunto de datos de entrenamiento, entregado por el `DataLoader` de la Parte 2. `input` es el lote de datos y `target` las etiquetas verdaderas correspondientes. | Sin iterar por lotes no hay datos que pasarle al modelo. |
| `optimizer.zero_grad()` | **Resetea los gradientes** de todos los parámetros del modelo. | Los gradientes de PyTorch se **acumulan** por diseño (se suman) en cada `backward()`. Si no se resetean, los gradientes de un batch se mezclan con los del batch anterior, y la actualización de pesos queda calculada sobre una mezcla de lotes distintos. No tira error: el entrenamiento simplemente aprende mal o de forma errática. |
| `outputs = model(input)` | Se calcula la salida dada una entrada de datos: es el **forward pass**, ejecuta el método `forward` del modelo. | Sin esto no hay predicción que comparar contra `target`; no se puede seguir. |
| `loss = criterion(outputs, target)` | Se calcula el **error (la pérdida)**, comparando la salida del modelo (`outputs`) contra el valor verdadero (`target`), usando la función de pérdida definida en la Parte 3. | Sin loss no hay una señal numérica de qué tan mal predice el modelo, y `backward()` no tendría de dónde partir. |
| `loss.backward()` | Se realiza la **retropropagación** (*backpropagation*): calcula el gradiente de la pérdida respecto de cada parámetro del modelo, usando diferenciación automática. | Sin este paso no existen gradientes calculados; `optimizer.step()` no tendría nada que aplicar y los pesos jamás se actualizarían. Tampoco tira error: la red no aprende y la pérdida se mantiene prácticamente constante entre épocas. |
| `optimizer.step()` | **Actualiza los parámetros del modelo**, usando los gradientes calculados por `backward()` y la regla del optimizador elegido (SGD, Adam, etc.). | Los gradientes se calculan pero nunca se aplican: los pesos quedan siempre iguales a su valor inicial. La red tampoco aprende, y de nuevo, sin ningún error visible. |

El patrón general para recordar el orden: primero se limpia (`zero_grad`), después se predice (`model(input)`), se mide el error (`criterion`), se calculan los gradientes (`backward`) y por último se actualizan los pesos (`step`). Ninguno de los cinco pasos, si se omite, produce un error de ejecución: el síntoma siempre es el mismo, una red que "corre" pero no mejora.

## 3. Evaluar un modelo

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```

Evaluar requiere determinar un conjunto de datos de **validación**: datos no utilizados en la etapa de entrenamiento, para evitar sesgos. El tiempo requerido para la evaluación es proporcional al tamaño del conjunto de datos de validación.

| Elemento | Qué hace |
|---|---|
| `model.eval()` | Pone el modelo en **modo evaluación**. Desactiva comportamientos que solo tienen sentido durante el entrenamiento, como `Dropout` (deja de "apagar" neuronas) o el uso de estadísticas de batch en `BatchNorm` (usa las estadísticas acumuladas en vez de las del lote actual). |
| `with torch.no_grad():` | Desactiva el cálculo de gradientes dentro del bloque. No hace falta retropropagar durante la evaluación, así que esto ahorra memoria y cómputo. |
| `torch.max(outputs.data, 1)` | Para cada muestra, devuelve el valor máximo y el **índice** de la clase con mayor puntaje a lo largo de la dimensión 1 (las clases). El `_` descarta el valor máximo y se queda con `predicted`, el índice, que es la clase predicha. |
| `total += target.size(0)` | Acumula manualmente la cantidad total de muestras evaluadas (el tamaño del batch). |
| `correct += (predicted == target).sum().item()` | Compara predicción contra etiqueta real elemento a elemento, suma cuántas coincidieron en el batch, y `.item()` extrae ese conteo como número de Python (en vez de tensor de un solo elemento). |
| `accuracy = 100 * correct / total` | Calcula el porcentaje de aciertos sobre el total de muestras evaluadas, acumulado a mano en vez de con una función de métricas ya armada. |

> **Nota:** este cálculo de exactitud es completamente manual (contadores `correct`/`total`), a diferencia de `scikit-learn`, donde `accuracy_score` hace lo mismo en una línea. Es intencional: en PyTorch no hay un objeto "modelo entrenado" con métodos de evaluación incorporados como en scikit-learn, así que ese conteo hay que escribirlo explícitamente cada vez.

## 4. Relación con el resto del módulo

- Este bucle es la culminación de las tres partes anteriores: los `DataLoader` de la Parte 2 entregan los lotes, el modelo y la loss/optimizador de la Parte 3 son los que se usan acá.
- El mecanismo de `zero_grad` / `backward` / `step` es la implementación concreta, en PyTorch, del descenso de gradiente descripto de forma general en [`06-fundamentos-de-redes-neuronales/teoria/Optimización.md`](../../06-fundamentos-de-redes-neuronales/teoria/Optimizaci%C3%B3n.md) (la regla `θ = θ − η · ∇θ J(θ)` es exactamente lo que aplica `optimizer.step()` usando el gradiente que calculó `loss.backward()`).
- El notebook [`OD_RN2_ESP_M02_S05_Implementación en PyTorch`](../notebooks/OD_RN2_ESP_M02_S05_Implementación%20en%20PyTorch%20-%20Recurso%20descargable.ipynb) implementa este mismo bucle de entrenamiento y evaluación sobre el dataset Iris, con los cinco pasos en el mismo orden.
