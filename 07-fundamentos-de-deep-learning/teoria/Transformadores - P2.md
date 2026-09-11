# Transformadores — Parte 2

> Conversión a Markdown de las slides del curso (Clase 24 — Transformadores, bloque 2). El PDF original está al lado.

## 1. AutoAtención (Self-Attention)

La **autoatención** es el mecanismo que le permite a cada palabra de una secuencia calcular una representación nueva de sí misma teniendo en cuenta a **todas** las demás palabras de la misma secuencia, ponderando cuánto le "importa" cada una.

Antes de entrar al mecanismo interno, la primera slide de este bloque recuerda cómo funcionaba la atención clásica sobre una RNN encoder-decoder (la vista en la clase anterior, [`Mecanismos de Atención.md`](Mecanismos%20de%20Atención.md)), en cinco pasos:

1. **Prepare inputs**: se tienen los estados ocultos del encoder $h_1, h_2, h_3$ y el estado oculto del decoder en el paso actual (paso 4 en el ejemplo).
2. **Score each hidden state**: se calcula un puntaje de compatibilidad entre el estado del decoder y cada estado del encoder (en el ejemplo: `13, 9, 9`).
3. **Softmax the scores**: los puntajes se normalizan con softmax para que sumen 1 (`0.96, 0.02, 0.02`).
4. **Multiply each vector by its softmaxed score**: cada estado del encoder se pondera por su score normalizado.
5. **Sum up the weighted vectors**: se suman los vectores ponderados, dando el **vector de contexto** para ese paso del decoder.

> **Nota:** esta slide es un repaso de la atención "clásica" (Bahdanau/Luong, aplicada sobre una RNN) que ya se vio en la clase anterior del módulo. Sirve de puente hacia la autoatención del Transformer: la diferencia central es que ahí la consulta salía del decoder RNN y las claves/valores del encoder RNN; en la autoatención del Transformer, consulta, claves y valores salen todos de la **misma** secuencia.

## 2. Las tres matrices: Query, Key, Value

La autoatención del Transformer generaliza el mecanismo anterior con tres proyecciones lineales aprendidas de cada embedding de entrada $x_i$:

| Vector | Nombre | Rol |
|---|---|---|
| $q_i = x_i W^Q$ | **Query** (consulta) | lo que esta palabra "busca" en las demás |
| $k_i = x_i W^K$ | **Key** (clave) | lo que esta palabra "ofrece" para ser encontrada |
| $v_i = x_i W^V$ | **Value** (valor) | el contenido real que se termina agregando si la palabra es relevante |

Las matrices $W^Q$, $W^K$ y $W^V$ son pesos que se aprenden durante el entrenamiento, y son las **mismas para todas las posiciones** de la secuencia (se aplican palabra por palabra, igual que la red feed forward del bloque anterior).

> **Nota — la analogía de la búsqueda:** pensar el mecanismo como un buscador de archivos. La Query es lo que se escribe en la barra de búsqueda; cada archivo tiene una Key (una especie de título o resumen contra el que se compara la búsqueda) y un Value (el contenido real del archivo). El buscador compara la Query contra todas las Keys, y cuanto más se parece una Key a la Query, más peso recibe el Value correspondiente en el resultado final. En autoatención, cada palabra hace de buscador y de archivo al mismo tiempo: genera su propia Query para consultar a las demás, y expone su propia Key/Value para que las demás la consulten a ella.

La slide lo ilustra con la oración de ejemplo `Thinking Machines`: cada palabra ($x_1$, $x_2$) genera su propio par $q$/$k$/$v$ multiplicando su embedding por las matrices compartidas $W^Q$, $W^K$, $W^V$.

## 3. El cálculo paso a paso

Con las matrices Q, K, V ya calculadas, el cómputo de la autoatención sigue estos pasos (ilustrados en la slide sobre `Thinking Machines`, calculando la salida para la posición de `Thinking`):

1. **Score**: se calcula el producto punto entre la Query de la palabra actual y la Key de cada palabra (incluida ella misma): `q1 · k1 = 112`, `q1 · k2 = 96`.
2. **Divide by √d_k**: los scores se dividen por la raíz cuadrada de la dimensión de las claves (`√d_k = 8` en el ejemplo, con $d_k = 64$): `112/8 = 14`, `96/8 = 12`.
3. **Softmax**: se aplica softmax a los scores escalados, dando pesos que suman 1: `0.88` y `0.12`.
4. **Softmax × Value**: cada Value se pondera por su peso de softmax.
5. **Sum**: se suman los Values ponderados, dando el vector de salida $z_1$ para esa posición (y análogamente $z_2$ para `Machines`).

En notación matricial, para toda la secuencia a la vez:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

## 4. Por qué se divide por √d_k

> **Nota:** ampliación no cubierta explícitamente en la slide más allá del cálculo numérico, necesaria para entender por qué el paso 2 existe.

El producto punto $Q \cdot K^T$ crece en magnitud a medida que crece la dimensión $d_k$ de los vectores: si cada componente de $q$ y $k$ tiene varianza del orden de 1, el producto punto de vectores de dimensión $d_k$ tiene varianza del orden de $d_k$. Con $d_k = 64$ (como en el ejemplo) o valores más altos usados en la práctica, los scores pueden volverse muy grandes.

Cuando los scores que entran al softmax son muy grandes en valor absoluto, el softmax se **satura**: la función empuja casi todo el peso hacia el score más alto y el resto queda prácticamente en cero. En esa zona de saturación, la derivada del softmax es casi nula, lo que deja **gradientes casi nulos** para las palabras que "perdieron" la comparación — el entrenamiento se vuelve muy lento o se estanca, porque el modelo casi no recibe señal para ajustar esos pesos.

Dividir por $\sqrt{d_k}$ reescala los scores para que su varianza vuelva a ser del orden de 1 sin importar cuán grande sea $d_k$, manteniendo al softmax en una zona donde sus gradientes siguen siendo informativos. Es la misma lógica de fondo que motiva normalizar entradas o usar inicializaciones de pesos cuidadosas en cualquier red: evitar zonas de saturación de las no linealidades.

## 5. Relación con el resto del módulo

- Este bloque es el núcleo de todo el resto de la Clase 24: [`Transformadores - P3.md`](Transformadores%20-%20P3.md) repite exactamente este mecanismo en paralelo con múltiples juegos de matrices Q/K/V (múltiples cabezales), y [`Transformadores - P6.md`](Transformadores%20-%20P6.md) lo reutiliza con una restricción adicional (enmascarado) en la autoatención del decoder, y con Q proveniente del decoder pero K/V del encoder en la atención cruzada.
- La comparación de la sección 1 con la atención vista en [`Mecanismos de Atención.md`](Mecanismos%20de%20Atención.md) es la misma progresión conceptual que already se vio dentro de la familia recurrente: así como la [GRU](<Unidades Recurrentes con Compuertas (GRU).md>) y la [LSTM](<Memoria a Largo Plazo (LSTM).md>) agregaron mecanismos (compuertas) sobre la RNN clásica para controlar el flujo de información, acá la autoatención generaliza y independiza a la atención de la necesidad de una RNN de por medio.
- El escalado por $\sqrt{d_k}$ es puramente numérico y no tiene equivalente directo en las arquitecturas recurrentes del módulo, porque ahí no existe un producto punto de esta escala entre vectores de consulta y de clave.
