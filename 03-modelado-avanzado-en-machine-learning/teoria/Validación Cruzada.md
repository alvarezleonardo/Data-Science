<!-- Convertido automáticamente desde: Validación Cruzada.pdf -->

Validación Cruzada

01 Introducción a la Validación Cruzada

La validación cruzada es una técnica utilizada para evaluar el rendimiento de
un modelo predictivo, estimar la capacidad de generalización del modelo a
datos no vistos, ayudando a prevenir el sobreajuste y proporcionando una
evaluación más fiable del modelo.

Tipos de Validación Cruzada

- K-Fold Cross-Validation: Se divide el conjunto de datos en k subconjuntos
  (folds). Se entrena el modelo k veces, cada vez usando un fold diferente
  como conjunto de prueba y el resto como conjunto de entrenamiento. Se
  promedian los resultados para obtener una métrica final.
- Leave-One-Out Cross-Validation (LOOCV): Cada observación se utiliza como
  conjunto de prueba una vez. Se entrena el modelo n veces (siendo n el número
  de observaciones). Muy intensivo computacionalmente.
- Stratified K-Fold Cross-Validation: Similar a K-Fold, pero preserva la
  proporción de clases en cada fold. Especialmente útil para problemas de
  clasificación con clases desbalanceadas.

02 Validación Cruzada en Regresión

La validación cruzada en regresión sigue los mismos principios que en
clasificación, pero enfocada en problemas de predicción continua.

Métricas comunes:

- Error Cuadrático Medio (MSE)
- Raíz del Error Cuadrático Medio (RMSE)
- Coeficiente de Determinación (R²)

Implementación de Validación Cruzada de Regresión

1. Dividir el conjunto de datos en k folds.
2. Entrenar el modelo en k-1 folds.
3. Validar el modelo en el fold restante.
4. Repetir para cada fold y calcular la métrica de rendimiento.
5. Promediar las métricas obtenidas.

¡Muchas gracias!
