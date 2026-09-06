# Optimización

> Conversión a Markdown de las slides del curso (unidad "Optimización y regularización", bloque 03). El PDF original está al lado.

> **Nota:** varias slides son imágenes sin texto (la superficie de pérdida en 3D, los tres escenarios de tasa de aprendizaje). Se describen abajo. El deck repite la misma slide agregando un párrafo por página; se consolidó.

## 1. Qué es optimizar

La **optimización** es el proceso de **ajustar los pesos y sesgos del modelo para minimizar una función de pérdida** (§35). Esa función cuantifica cómo está realizando el modelo sus predicciones en comparación con los valores verdaderos.

El **descenso de gradiente** es el método iterativo que se usa para minimizarla. Se calcula el **gradiente** como la derivada de la función de error **con respecto a todos los parámetros de la red**.

La slide lo ilustra con una superficie en 3D con forma de cuenco: los ejes horizontales son dos parámetros, la altura es la pérdida, y el vector marcado sobre la superficie es la dirección del gradiente. Entrenar es bajar por ese cuenco hasta el fondo.

## 2. La regla de actualización

Una vez obtenido el vector gradiente, se actualizan los parámetros **restando a su valor actual el valor del gradiente correspondiente, multiplicado por una tasa de aprendizaje**:

```
θ = θ − η · ∇θ J(θ)
```

| Símbolo | Qué es |
|---|---|
| `θ` | los **parámetros** (pesos y sesgos) |
| `η` | la **tasa de aprendizaje** (*learning rate*) |
| `∇θ J(θ)` | el **gradiente de la función de pérdida** con respecto a `θ` |

El signo menos es lo importante: el gradiente apunta en la dirección de **mayor crecimiento** de la pérdida, así que para minimizar hay que moverse en la dirección **contraria**.

## 3. La tasa de aprendizaje

> "Es muy importante una elección adecuada de la tasa de aprendizaje."

La slide muestra tres parábolas con el recorrido del optimizador dibujado en rojo:

| Escenario | Qué se ve en la slide | Consecuencia |
|---|---|---|
| **η demasiado chica** | muchos pasitos cortos bajando por la curva | converge, pero **lento**: puede agotar `max_iter` antes de llegar |
| **η adecuada** | pocos pasos que llegan al fondo | converge **rápido y estable** |
| **η demasiado grande** | saltos que cruzan el valle de un lado al otro, alejándose | **diverge**: la pérdida oscila o explota en vez de bajar |

Es el hiperparámetro más sensible del entrenamiento. En `scikit-learn` se controla con `learning_rate_init`, y `learning_rate='adaptive'` permite que baje sola cuando la pérdida deja de mejorar.

## 4. Las tres formas de hacer descenso por gradiente

El descenso por gradiente puede realizarse de tres modos, según **cada cuánto se actualizan los pesos**:

| Modo | Cuándo actualiza | Característica |
|---|---|---|
| **Estocástico (SGD)** | cada vez que **una muestra** de entrenamiento se evalúa por la red | actualizaciones muy frecuentes y ruidosas |
| **En lotes (batch)** | solo cuando **concluye una época**, es decir, cuando terminó de evaluarse todo el conjunto de entrenamiento | actualizaciones estables pero pocas, y caras en memoria |
| **En mini-lotes (mini-batch)** | al finalizar cada **minilote**, usando la **media del gradiente** a lo largo de ese lote | el compromiso entre los dos: es lo que se usa en la práctica |

En `scikit-learn`, `MLPClassifier` con `solver='sgd'` o `'adam'` trabaja en **mini-lotes**, y el tamaño se fija con `batch_size`. El nombre "sgd" es histórico: no actualiza de a una muestra salvo que se ponga `batch_size=1`.

## 5. Relación con el resto del módulo

- Lo que se minimiza es la **función de pérdida** (§35); acá se ve **cómo** se minimiza.
- La **regla delta** del perceptrón (§28) es este mismo mecanismo en su forma más simple, con un solo peso por entrada y sin capas ocultas.
- **Backpropagation** (§31) es el algoritmo que calcula eficientemente ese gradiente `∇θ J(θ)` para todas las capas, aplicando la regla de la cadena hacia atrás.
- La **saturación** de las activaciones (§32) es un problema de este proceso: si la derivada de la activación es casi 0, el gradiente que llega a las capas iniciales se apaga.
