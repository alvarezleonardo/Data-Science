# Introducción a PyTorch - Parte 1

> Conversión a Markdown de las slides del curso (Clase 4 — Introducción a PyTorch, bloque 1). El PDF original está al lado.

## 1. Qué es PyTorch

PyTorch es una biblioteca de código abierto para el aprendizaje automático, especialmente diseñada para el desarrollo y entrenamiento de modelos de **deep learning**. Es conocida por su simplicidad y facilidad de uso, así como por su integración transparente con el ecosistema de Python.

- Desarrollada por **Facebook**. Se convirtió en código abierto en **2017** y ha estado bajo la administración de la **PyTorch Foundation** desde **2022**.
- Una de sus características más destacadas es su enfoque en la **programación dinámica**.
- Dispone de soporte para su ejecución en tarjetas gráficas (**GPU**).
- Integración con la comunidad científica y académica.
- Cuenta con un soporte sólido para la **diferenciación automática**.
- Proporciona un conjunto completo de herramientas para la creación y entrenamiento de modelos de aprendizaje profundo.

### Grafo dinámico vs. grafo estático

La slide no lo desarrolla, pero "programación dinámica" es la distinción conceptual clave entre PyTorch y TensorFlow:

| | Grafo dinámico (PyTorch) | Grafo estático (TensorFlow clásico, `tf.Graph`) |
|---|---|---|
| Cuándo se construye | en cada `forward`, sobre la marcha (*define-by-run*) | una vez, antes de correr los datos (*define-then-run*) |
| Debugging | se puede usar `print`, breakpoints y el debugger de Python línea por línea | requiere herramientas propias (sesiones, `tf.function` trazado) |
| Flexibilidad | admite control de flujo con `if`/`for` de Python dependiente de los datos | el control de flujo debe expresarse con operaciones del grafo (`tf.cond`, `tf.while_loop`) |
| Costo | recompila el grafo en cada paso, algo más lento de optimizar | el grafo se optimiza y compila una sola vez, potencialmente más rápido en producción |

> **Nota:** TensorFlow 2.x adoptó *eager execution* por defecto, que también es dinámica; la diferencia histórica sigue siendo la razón por la que PyTorch se asocia a "investigación" y TensorFlow a "producción", aunque hoy ambos ofrecen los dos modos.

## 2. Tensores

En cualquier algoritmo de aprendizaje automático, los datos deben representarse numéricamente. En PyTorch, esto se logra a través de **tensores**, que sirven como unidades fundamentales de datos utilizadas para el cálculo.

- Un tensor es una **generalización de matrices a más dimensiones**.
- Disponen de soporte para su ejecución en tarjetas gráficas (**GPU**).
- Funcionan de manera similar a los `ndarray` que se usan en **NumPy**.
- Soportan una amplia gama de **tipos de datos**, incluyendo enteros, puntos flotantes y booleanos, lo que permite adaptarse a diferentes necesidades y contextos.
- Capacidad para realizar operaciones avanzadas, como transposiciones, indexación, cortes y operaciones aritméticas.

En la práctica, un tensor de PyTorch es lo mismo que un array de NumPy, pero con dos capacidades extra: puede vivir en GPU (`.to(device)`) y PyTorch puede rastrear las operaciones hechas sobre él para calcular gradientes automáticamente (`requires_grad=True`), que es la base de `loss.backward()` (ver Parte 4).

## 3. Relación con el resto del módulo

- Esta parte es la introducción conceptual; la Parte 2 usa tensores concretos al cargar datos con `torchvision` y `DataLoader`.
- La diferenciación automática mencionada acá es lo que hace posible `loss.backward()` en el bucle de entrenamiento de la Parte 4.
- Para la configuración del entorno en esta máquina (instalación de `torch`, verificación de aceleración por GPU/MPS) ver [Configuración del entorno.md](Configuración%20del%20entorno.md).
- El flujo completo (datos, modelo, pérdida, optimizador, entrenamiento y evaluación) está implementado sobre el dataset Iris en el notebook [`OD_RN2_ESP_M02_S05_Implementación en PyTorch - Recurso descargable.ipynb`](../notebooks/OD_RN2_ESP_M02_S05_Implementación%20en%20PyTorch%20-%20Recurso%20descargable.ipynb).
