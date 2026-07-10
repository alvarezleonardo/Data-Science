<!-- Convertido automáticamente desde: Más Práctica 2.pdf -->

Más Práctica 2

Ejercicio

El dataset `diamonds.csv` contiene los precios y otros atributos de casi **54.000
diamantes**.

Sus features son:
- **price:** precio en USD (326–18.823). **Es la variable target.**
- **carat:** peso del diamante (0.2–5.01).
- **cut:** calidad del corte (Fair, Good, Very Good, Premium, Ideal).
- **color:** color, de J (peor) a D (mejor).
- **clarity:** claridad: I1 (peor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (mejor).
- **x:** largo en mm (0–10.74).
- **y:** ancho en mm (0–58.9).
- **z:** profundidad en mm (0–31.8).
- **depth:** profundidad total = z / mean(x, y) = 2·z / (x + y) (43–79).
- **table:** ancho del tope relativo al punto más ancho (43–95).

Nota sobre el escalado: en este ejercicio se escalan las features con
`MinMaxScaler` (para tener un ejemplo con una alternativa a `StandardScaler`), no
porque rinda mejor. En general la **estandarización** suele ser más práctica: al
centrar en media 0 y desvío 1, las columnas adoptan forma de distribución normal
(facilita el aprendizaje de los pesos, que se inicializan cerca de 0) y es menos
sensible a outliers que el min-max.

Consignas:
1. Normalizar las features y crear las **dummies** necesarias para predecir
   `price`.
2. Separar en **train / test**.
3. Ajustar una **regresión lineal múltiple** con `statsmodels` y evaluar la
   significancia de cada coeficiente.
4. Ajustar con **regularización Lasso + validación cruzada** para estimar el mejor
   `α`. ¿Cuál es el mejor `α`? ¿Cuál es el R² en entrenamiento?
5. Ajustar con Lasso para ese `α` usando `statsmodels`; comparar con
   **scatterplots** los coeficientes (OLS vs Lasso) y los residuos.
6. Calcular la performance en **test** con `statsmodels` y `scikit-learn`, y
   comparar ambas usando **MAE** y **RMSE**.
