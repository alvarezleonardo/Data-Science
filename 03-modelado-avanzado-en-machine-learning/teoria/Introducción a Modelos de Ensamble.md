<!-- Convertido automáticamente desde: Introducción a Modelos de Ensamble.pdf -->

Introducción a Modelos de Ensamble

01 ¿Por qué Ensambles?

La idea es entrenar muchos modelos y hacerlos votar. La clasificación
resultante es la que reciba más votos. Si todos los modelos son muy parecidos,
no van a agregar mucha información nueva en la votación. Necesitamos modelos
diferentes entre sí, poco correlacionados.

¿Qué es un modelo de ensamble?

Los modelos de ensamble combinan las decisiones de varios modelos para mejorar
el rendimiento general.

Principales causas de error:
- Ruido
- Sesgo
- Varianza

Beneficios de los modelos de ensamble:
- Ayudan a minimizar el ruido, el sesgo y la varianza.
- Mejoran la estabilidad y precisión de los algoritmos de aprendizaje automático.
- La combinación de resultados de múltiples modelos suele ser más precisa que el
  uso de un solo modelo.

Estrategia:
- Los ensambles emplean un enfoque "divide and conquer" para mejorar la
  performance.

02 Técnicas de Ensamble

Familias de Métodos de Ensamble

Métodos de Averaging (basados en promedios): construyen varios estimadores de
forma independiente y luego promedian sus predicciones.
- Ventaja: el modelo resultante suele ser mejor que cualquier estimador base
  separado.
- Ejemplos: Bagging, Random Forest.

Métodos de Boosting: los estimadores base se construyen secuencialmente,
tratando de reducir el sesgo del estimador combinado.
- Ventaja: se centran en mejorar aquellos casos con peor performance.
- Objetivo: combinar varios modelos débiles para producir un ensamble potente.
- Ejemplos: AdaBoost, Gradient Tree Boosting.

El espacio de Hipótesis

En cualquier tarea de aprendizaje supervisado, el objetivo es hacer predicciones
de la verdadera función de clasificación `f` aprendiendo el clasificador `h`.
Proceso: buscar en un espacio de hipótesis `H` la función más apropiada que
describa la relación entre las características (features) y el objetivo.

Problemas comunes en clasificadores base:
1. Estadísticos: limitaciones en los datos que afectan la precisión.
2. Computacionales: restricciones en los recursos computacionales disponibles.
3. De representación: dificultad en representar adecuadamente la función de
   clasificación real.

Uso de la técnica de ensamble con modelos distintos

Meta-modelos basados en árboles de decisión:
- Bagging: Random Forest.
- Boosting: Gradient Boost, ADA Boost, XG Boost.
