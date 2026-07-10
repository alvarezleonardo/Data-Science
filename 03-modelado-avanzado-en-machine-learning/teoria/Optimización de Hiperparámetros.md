<!-- Convertido automáticamente desde: Optimización de Hiperparámetros.pdf -->

Optimización de Hiperparámetros

01 ¿Qué son los Hiperparámetros?

Parámetros que se establecen **antes** del proceso de entrenamiento y **no** se
ajustan durante el entrenamiento. Determinan la estructura del modelo y cómo se
entrenará. Afectan directamente el rendimiento y la capacidad de generalización;
un ajuste incorrecto puede llevar a **overfitting** o **underfitting**.

Principales hiperparámetros por modelo

Random Forest:
- **n_estimators:** número de árboles del bosque. Más árboles mejoran la precisión
  pero aumentan el coste computacional.
- **max_depth:** profundidad máxima de cada árbol. Valores altos → sobreajuste.
- **min_samples_leaf:** mínimo de muestras por hoja. Evita hojas con muy pocas
  muestras, reduciendo el sobreajuste.
- **max_features:** número de características a considerar en cada división. Menos
  características → más variedad entre árboles y menos sobreajuste.

SVM:
- **C (regularización):** rigidez del margen. Alto → margen estrecho y sobreajuste;
  bajo → margen amplio y subajuste.
- **gamma:** influencia de un solo punto de entrenamiento. Alto → sobreajuste.
- **kernel:** lineal, polinomial, RBF, sigmoide.

Boosting (Gradient Boosting):
- **n_estimators:** número de árboles secuenciales. Muchos → posible sobreajuste.
- **learning_rate:** reducción aplicada a cada árbol. Tasa baja necesita más
  árboles pero generaliza mejor.
- **max_depth:** profundidad de cada árbol.
- **subsample:** fracción de muestras por árbol. Submuestrear reduce varianza y
  sobreajuste.

02 Búsqueda de hiperparámetros

Los hiperparámetros son cruciales para el rendimiento. **Grid Search** y **Random
Search** son técnicas útiles para encontrar los hiperparámetros óptimos. Ajustarlos
bien puede mejorar significativamente la precisión y la generalización del modelo.
