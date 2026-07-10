<!-- Convertido automáticamente desde: Introducción al Descenso del Gradiente.pdf -->

Introducción al Descenso del Gradiente

01 ¿Qué es gradiente?

El gradiente es la **derivada o tasa de cambio** de una función. Es un vector (una
dirección para moverse) que:
- Apunta en la dirección de **mayor aumento** de una función.
- Es **cero** en un máximo o un mínimo local (porque no hay una única dirección de
  aumento).

El término "gradiente" se usa normalmente para funciones con **varias variables** y
una única salida (un campo escalar). Para funciones de una sola variable, hablar de
gradiente (en vez de pendiente/derivada) es innecesariamente confuso.

02 ¿Qué es descenso del gradiente?

Es un algoritmo de optimización **genérico** e **iterativo** para encontrar el
mínimo de una función: comenzamos en un punto aleatorio y nos movemos en la
**dirección negativa del gradiente** para alcanzar los mínimos locales / globales.

En machine learning buscamos el mínimo de la **función de costo** (J) o función de
pérdida, que mide la diferencia entre el valor real y la predicción del modelo.

¿Cuándo usamos descenso del gradiente?
- Cuando los parámetros del modelo **no se pueden calcular analíticamente** (por
  ejemplo, con álgebra lineal) y deben buscarse con un algoritmo de optimización.
- Cuando el modelo tiene muchas variables/observaciones, o su función de costo no
  tiene solución analítica (como en la regresión logística y las redes neuronales).

Función de costo
- La función de costo de la regresión lineal es **convexa**, lo que garantiza que
  el descenso del gradiente converja al mínimo global.
- En funciones de costo con más de un mínimo (como la de la función logística),
  conviene iniciar el algoritmo desde **diferentes posiciones** para que no quede
  "atrapado" en un mínimo local.

Learning rate
- La tasa de aprendizaje (learning rate) `α` es un hiperparámetro muy importante:
  - **Grande:** puede oscilar y no converger (saltar de un lado al otro del valle).
  - **Chico:** converge, pero lento.
- Para reducir el tiempo de convergencia conviene que las variables estén
  **normalizadas**.
