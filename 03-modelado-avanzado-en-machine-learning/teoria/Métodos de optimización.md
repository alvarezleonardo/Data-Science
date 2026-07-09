<!-- Convertido automáticamente desde: Métodos de optimización.pdf -->

Tipos de Optimización en Machine Learning

01 Introducción a la Optimización en Machine Learning

Proceso de ajustar los parámetros del modelo para **minimizar (o maximizar) una
función objetivo**. La optimización es crucial para encontrar el modelo más
adecuado que se ajuste a los datos y generalice bien en datos nuevos. Se usa en
la mayoría de los algoritmos de ML (regresión, clasificación, clustering, redes
neuronales, etc.).

Optimización Convexa

Un problema es convexo si la función objetivo es convexa (cualquier línea recta
entre dos puntos del dominio queda por encima de la función).
- Características: un **único mínimo global**. Técnicas de optimización eficientes
  y garantizadas para encontrarlo.
- Ejemplo: regresión lineal, SVM lineal.

Optimización No Convexa

Un problema es no convexo si la función objetivo no es convexa, pudiendo tener
**múltiples mínimos locales** y máximos.
- Características: puede tener múltiples mínimos locales; más compleja y desafiante
  para optimizar.
- Ejemplo: redes neuronales profundas, SVM no lineal.

02 Comparación: Convexa vs No Convexa

Convexa:
- Facilidad de encontrar el mínimo global.
- Algoritmos más rápidos y eficientes.
- Usada en problemas donde la función de pérdida es simple y bien entendida.

No Convexa:
- Mayor flexibilidad y capacidad para capturar relaciones complejas.
- Desafiante por la presencia de múltiples mínimos locales.
- Requiere técnicas avanzadas y a menudo heurísticas para la optimización.
