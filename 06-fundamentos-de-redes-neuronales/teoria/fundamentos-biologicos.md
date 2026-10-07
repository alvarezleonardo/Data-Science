# Fundamentos Biológicos

> Conversión a Markdown de la slide del curso. El PDF original está en [`material/`](material/).

## 1. Inspiración biológica de las redes neuronales artificiales

Las **redes neuronales artificiales (RNA)** están inspiradas en la estructura y el funcionamiento del cerebro humano. El cerebro está compuesto por aproximadamente **86 mil millones de neuronas**, que se comunican entre sí a través de conexiones llamadas **sinapsis**.

Cada neurona **recibe señales** de otras neuronas, **procesa** esa información y **transmite** señales a sus neuronas vecinas. Esta red de comunicación masiva es la que permite que el cerebro realice tareas complejas.

### Partes de la neurona biológica

- **Dendritas:** reciben las señales de entrada provenientes de otras neuronas.
- **Cuerpo celular (soma):** contiene el núcleo y procesa las señales recibidas.
- **Núcleo:** centro de control de la célula.
- **Axón:** transmite la señal de salida hacia otras neuronas.

## 2. De la neurona biológica a la neurona artificial

Las RNA emulan esta estructura biológica mediante unidades de procesamiento simples llamadas **neuronas artificiales**. Cada neurona artificial recibe múltiples entradas, las procesa aplicando una **función de activación** y produce una salida.

El paralelismo entre ambos modelos es directo:

| Elemento biológico | Elemento artificial |
|---|---|
| Dendritas | Entradas (x₁, x₂, ..., xₙ) |
| Fuerza de la sinapsis | Pesos (w₁, w₂, ..., wₙ) |
| Cuerpo celular (integración de señales) | Sumatoria ponderada (Σ) |
| Umbral de disparo de la neurona | Función de activación (Φ) |
| Axón (señal transmitida) | Salida (y) |

En el modelo artificial, cada entrada `x` se multiplica por su peso `w` correspondiente; la neurona suma todos esos productos y aplica una función de activación `Φ` sobre el resultado para producir la salida `y`.

### Organización en capas

Al igual que las neuronas biológicas se organizan en redes, las neuronas artificiales se organizan en **capas**:

- **Capa de entrada:** recibe los datos iniciales.
- **Capas ocultas** (una o más): procesan la información intermedia.
- **Capa de salida:** produce el resultado final del modelo.

## 3. Aprendizaje: paralelismo y diferencias

**Similitud:** el aprendizaje en el cerebro humano ocurre a través del fortalecimiento o debilitamiento de las conexiones sinápticas entre neuronas. De forma análoga, las RNA aprenden **ajustando los parámetros (pesos)** que existen entre las conexiones de las neuronas.

**Diferencia clave — eficiencia energética:** mientras que el cerebro humano realiza sus funciones utilizando señales electroquímicas y consume relativamente **poca energía**, las RNA —especialmente las más profundas y complejas— requieren un **poder computacional y energético considerable**.

> **Nota:** el modelo biológico es una fuente de inspiración conceptual para las RNA, no una réplica exacta. El mecanismo de aprendizaje (ajuste de pesos vía backpropagation y optimización numérica) es matemáticamente muy distinto de la plasticidad sináptica real, aunque ambos comparten la idea de "fortalecer o debilitar conexiones" como base del aprendizaje.
