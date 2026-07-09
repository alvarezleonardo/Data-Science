<!-- Convertido automáticamente desde: Introducción a Boosting - Gradient Boosting.pdf -->

Gradient Boosting

- Creador: Jerome H. Friedman.
- Año: 1999.
- Importancia: un método eficaz y flexible, ampliamente adoptado en la industria.

Cada árbol se entrena sobre los residuales del anterior:
`r₁ = y₁ − ŷ₁`, `r₂ = r₁ − r̂₁`, `r₃ = r₂ − r̂₂`, … ponderados por el learning
rate η.

Funcionamiento de Gradient Boosting:
- Inicialización: comienza con un modelo inicial (generalmente uno trivial como
  la media).
- Iteración: entrenar un nuevo modelo para predecir los residuos (errores) del
  modelo actual. Actualizar el modelo sumando el nuevo modelo ponderado por un
  factor de aprendizaje.
- Combinación: repetir el proceso hasta alcanzar un número de iteraciones o un
  umbral de error.

Ventajas:
- Precisión: puede producir modelos con alta precisión.
- Flexibilidad: puede optimizar diferentes funciones de pérdida.
- Regularización: incluye métodos para prevenir el sobreajuste.

Desventajas:
- Tiempo de entrenamiento: puede ser lento debido a su naturaleza iterativa.
- Complejidad: mayor complejidad en comparación con otros métodos de boosting.
- Sensibilidad a parámetros: requiere ajuste cuidadoso de hiperparámetros.

Aplicaciones de Gradient Boost:
- Previsión de ventas: predicción de tendencias y ventas futuras.
- Detección de fraude: identificación de transacciones fraudulentas.
- Sistemas de recomendación: personalización de recomendaciones de productos.
