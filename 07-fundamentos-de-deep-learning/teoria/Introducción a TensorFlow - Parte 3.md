# Introducción a TensorFlow - Parte 3

> Conversión a Markdown de las slides del curso (Clase 3 — Introducción a TensorFlow, bloque 3). El PDF original está en [`material/`](material/).

## 1. Entrenar un modelo

`model.fit` es el método de Keras que realiza el entrenamiento del modelo. Toma los datos de entrada, las etiquetas, y otros parámetros para controlar el proceso de entrenamiento.

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
```

| Parámetro | Qué controla |
|---|---|
| `X_train, y_train` | Datos y etiquetas de entrenamiento. |
| `epochs` | Cuántas veces se recorre el conjunto de entrenamiento completo. |
| `batch_size` | Tamaño del mini-lote con el que se actualizan los pesos en cada paso (ver `Optimización.md`, módulo 06: descenso por mini-lotes). |
| `validation_data` | Datos que **no** se usan para ajustar pesos, sino para medir el desempeño del modelo al final de cada época y detectar sobreajuste. |

### El objeto `history`

`model.fit` no solo entrena: **devuelve** un objeto `History` cuyo atributo `.history` es un diccionario con la evolución de la pérdida y las métricas configuradas en `compile`, una entrada por época:

```python
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```

Con eso se puede graficar la curva de entrenamiento, la forma más directa de ver si el modelo converge, se estanca o sobreajusta:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='pérdida (train)')
plt.plot(history.history['val_loss'], label='pérdida (validación)')
plt.xlabel('época')
plt.ylabel('pérdida')
plt.legend()
plt.show()
```

Si la curva de `val_loss` deja de bajar (o empieza a subir) mientras `loss` sigue bajando, es la señal clásica de sobreajuste (ver `Regularización.md`, módulo 06).

## 2. Evaluar un modelo

`model.evaluate` evalúa el rendimiento del modelo entrenado utilizando un conjunto de datos de prueba (o validación) que el modelo **no ha visto** durante el entrenamiento.

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
```

> **Nota:** `model.evaluate` devuelve una lista con la pérdida seguida de cada métrica configurada en `compile`, en el mismo orden — acá se puede desempaquetar en dos variables porque `compile` solo tenía `metrics=['accuracy']`. Si se agregan más métricas, hay que desempaquetar la cantidad correspondiente de valores.

## 3. Relación con el resto del módulo

- Cierra el flujo completo de TensorFlow/Keras que arrancó en la **Parte 1** (conceptos) y siguió en la **Parte 2** (datos, arquitectura, `compile`): cargar datos → definir modelo → compilar → **entrenar (`fit`)** → **evaluar (`evaluate`)**.
- La curva de `loss` vs `val_loss` es la versión práctica, con TensorFlow, de la misma idea de **Gestión de modelos.md** (módulo 06) sobre monitorear el entrenamiento y decidir cuándo un modelo está listo o sobreajustado.
- Este mismo flujo (`fit` / `evaluate`) se retoma con otra sintaxis en las slides de **Introducción a PyTorch** del propio módulo 07, donde el mismo proceso se arma a mano con el *training loop* en vez de con un método único.
