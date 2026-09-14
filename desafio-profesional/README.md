# Desafío Profesional — Data Science

El Desafío Profesional es el proyecto integrador de la certificación: **un único trabajo de punta a punta**, en cuatro etapas, aplicado a un mismo caso de negocio elegido al comenzar. El objetivo es recorrer el pipeline completo de un proyecto de Data Science — desde la exploración inicial de los datos hasta la comparación entre Machine Learning "tradicional" y Deep Learning — documentando y justificando cada decisión.

El campus reparte estas cuatro etapas en tres módulos del programa, con una numeración que **no coincide** con la numeración de las etapas. Esta carpeta unifica las cuatro consignas y los casos de negocio en un solo lugar; el material de cada módulo (teoría, notebooks, datasets propios) sigue viviendo en su carpeta numerada.

## Equivalencia etapas ↔ módulos

| Etapa real (según los PDF) | Módulo del campus |
|---|---|
| Etapa 1 — Exploración Visual de los Datos (EDA) | [`02-desafio-profesional-etapas-1-y-2/`](../02-desafio-profesional-etapas-1-y-2/) |
| Etapa 2 — Limpieza y Transformación de Datos | [`02-desafio-profesional-etapas-1-y-2/`](../02-desafio-profesional-etapas-1-y-2/) |
| Etapa 3 — Modelado con Machine Learning | [`05-desafio-profesional-etapa-3/`](../05-desafio-profesional-etapa-3/) |
| Etapa 4 — Implementación de Redes Neuronales | [`09-desafio-profesional-etapa-4/`](../09-desafio-profesional-etapa-4/) |

## Casos de negocio disponibles

Se elige **uno solo** al empezar la Etapa 1, y se mantiene el mismo durante las cuatro etapas.

| Caso | Dominio | Tipo de problema |
|------|---------|-------------------|
| [subtes](casos-de-negocio/subtes/) | Movilidad urbana (CABA) | Series temporales / demanda |
| [airbnb](casos-de-negocio/airbnb/) | Precios de alojamiento | Regresión de precio |
| [cambio-climatico](casos-de-negocio/cambio-climatico/) | Ambiental | Regresión / tendencias |
| [diabetes](casos-de-negocio/diabetes/) | Salud | Clasificación binaria |

> Los datasets de cada caso (`*.zip`) no se versionan por superar el límite de 100 MB de GitHub. Se mantienen localmente / en Drive.

## Cómo se encadenan las etapas

Cada etapa consume el entregable de la anterior — no son trabajos independientes:

```
Etapa 1 (EDA)  →  Etapa 2 (limpieza/ETL)  →  Etapa 3 (modelado ML)  →  Etapa 4 (redes neuronales)
   dataset          dataset limpio            modelos + métricas       comparación ML vs. DL
   explorado         y transformado             de referencia          + conclusiones finales
```

- La **Etapa 1** entrega hallazgos y visualizaciones iniciales sobre el dataset crudo.
- La **Etapa 2** toma esos hallazgos y entrega el **dataset final transformado**, limpio de faltantes/outliers y con las variables listas para modelar.
- La **Etapa 3** toma ese dataset transformado y entrega uno o más **modelos de Machine Learning** entrenados y evaluados, con el modelo final justificado.
- La **Etapa 4** toma el mismo dataset y el modelo de referencia de la Etapa 3, y entrega una **red neuronal** que resuelve el mismo problema, comparando su desempeño contra los modelos tradicionales.

## Etapa 1 — Exploración Visual de los Datos (EDA)

**Objetivo:** familiarizarse con el dataset elegido, entender su estructura y aplicar técnicas de EDA (Pandas, Matplotlib, Seaborn) para descubrir patrones e insights preliminares.

**Entregable:** análisis exploratorio documentado (estadísticas descriptivas, outliers, patrones), scripts de Python y visualizaciones.

Consigna: [`consignas/etapa-1-exploracion-visual.md`](consignas/etapa-1-exploracion-visual.md)

Checklist:
- [ ] Dimensiones del dataset, tipos de variable y memoria (`df.info()`).
- [ ] Estadística descriptiva (`df.describe()`) y distribución de la variable target.
- [ ] Valores faltantes: `df.isnull().sum()` por columna.
- [ ] Outliers: detección visual (boxplots) y con `df.describe()`.
- [ ] Histogramas, scatter plots y heatmap de correlación.
- [ ] Documento de hallazgos con visualizaciones que los respalden.

## Etapa 2 — Limpieza y Transformación de Datos

**Objetivo:** asegurar la calidad de los datos — resolver faltantes, outliers e inconsistencias — y dejarlos transformados y documentados (proceso ETL) para el modelado posterior.

**Entregable:** dataset final transformado, scripts de limpieza (SQL + Python) y documentación del proceso ETL (Extract, Transform, Load).

Consigna: [`consignas/etapa-2-limpieza-y-transformacion.md`](consignas/etapa-2-limpieza-y-transformacion.md)

Checklist:
- [ ] Identificación y tratamiento de valores faltantes (`dropna()` / `fillna()`).
- [ ] Detección y tratamiento de outliers y duplicados.
- [ ] Normalización/escalado de variables numéricas (Min-Max o Z-score).
- [ ] Transformación de variables (fechas con `pd.to_datetime()`, agregaciones con `groupby()`).
- [ ] Documentación ETL: qué se extrajo, cómo se transformó y por qué, cómo quedó cargado.
- [ ] Dataset final verificado: consistente y sin errores.

## Etapa 3 — Modelado con Machine Learning

**Objetivo:** aplicar técnicas de Machine Learning (Scikit-Learn) sobre el dataset transformado para construir modelos predictivos que resuelvan el problema del caso de negocio, comparando varios algoritmos.

**Entregable:** scripts de modelado, informe de evaluación de cada modelo con métricas y justificación del modelo final.

Consigna: [`consignas/etapa-3-modelado-machine-learning.md`](consignas/etapa-3-modelado-machine-learning.md)

Checklist:
- [ ] Selección de modelos candidatos (supervisados y/o no supervisados según el problema).
- [ ] Split entrenamiento/prueba y validación cruzada.
- [ ] Evaluación con métricas adecuadas (precisión/recall/F1 para clasificación; MAE/RMSE para regresión).
- [ ] Optimización de hiperparámetros (`GridSearchCV`).
- [ ] Comparación entre modelos y selección justificada del modelo final.
- [ ] Documentación del proceso completo de modelado.

## Etapa 4 — Implementación de Redes Neuronales

**Objetivo:** aplicar Deep Learning (TensorFlow/Keras o PyTorch) sobre el mismo dataset y problema, y comparar el desempeño de la red neuronal contra los modelos de ML tradicionales de la Etapa 3.

**Entregable:** script de implementación de la red, informe comparativo Etapa 3 vs. Etapa 4, video explicativo (5 minutos) y documentación del proceso.

Consigna: [`consignas/etapa-4-implementacion-redes-neuronales.md`](consignas/etapa-4-implementacion-redes-neuronales.md)

Checklist:
- [ ] Diseño de arquitectura (MLP, CNN o RNN según el caso) — capas, neuronas, funciones de activación.
- [ ] Entrenamiento con validación (batches, optimizador Adam/SGD) evitando sobreajuste.
- [ ] Ajuste de hiperparámetros (tasa de aprendizaje, épocas, Dropout, Batch Normalization).
- [ ] Evaluación con métricas equivalentes a las de la Etapa 3, para poder comparar.
- [ ] Informe comparativo ML (Etapa 3) vs. Deep Learning (Etapa 4), con gráficos.
- [ ] Video explicativo de 5 minutos y documentación del proceso completo.

[← Volver al índice del repositorio](../README.md)
