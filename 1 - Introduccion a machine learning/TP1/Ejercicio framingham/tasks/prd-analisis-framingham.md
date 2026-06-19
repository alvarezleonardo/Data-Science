# PRD: Análisis Completo Dataset Framingham y Predicción de Enfermedad Coronaria

## Introducción

Completar el análisis exploratorio del dataset Framingham Heart Study y desarrollar un modelo de regresión logística para predecir el riesgo de enfermedad coronaria a 10 años (TenYearCHD). El proyecto debe responder preguntas específicas sobre el dataset, analizar correlaciones, proporciones de variables categóricas, tratar outliers mediante transformaciones, y evaluar el modelo con métricas completas incluyendo feature importance.

## Objetivos

- Responder de forma completa y documentada todas las preguntas planteadas sobre el dataset
- Analizar correlaciones entre variables numéricas y con la variable objetivo
- Visualizar y analizar proporciones de variables categóricas/binarias
- Aplicar transformaciones para reducir el impacto de outliers en variables continuas
- Entrenar un modelo de regresión logística para predecir TenYearCHD
- Evaluar el modelo con F1-Score, matriz de confusión completa y métricas adicionales
- Analizar feature importance y contribución de cada variable al modelo
- Mantener el estilo del notebook: markdown explicativo, ejemplos y documentación clara

## User Stories

### US-001: Análisis de Correlación entre Variables
**Descripción:** Como científico de datos, necesito analizar las correlaciones entre todas las variables numéricas para identificar relaciones lineales y multicolinealidad.

**Acceptance Criteria:**
- [ ] Calcular matriz de correlación completa (método Pearson)
- [ ] Crear heatmap de correlación con valores numéricos anotados
- [ ] Identificar y documentar correlaciones fuertes (|r| > 0.7)
- [ ] Analizar correlación de cada variable con TenYearCHD (variable objetivo)
- [ ] Agregar celda markdown explicando qué es correlación y cómo interpretarla
- [ ] Incluir interpretación de hallazgos principales

### US-002: Análisis de Proporciones de Variables Categóricas
**Descripción:** Como analista, necesito visualizar y cuantificar las proporciones de todas las variables categóricas y binarias del dataset.

**Acceptance Criteria:**
- [ ] Identificar todas las variables categóricas/binarias (male, education, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes, TenYearCHD)
- [ ] Crear gráficos de barras mostrando las proporciones de cada variable
- [ ] Calcular y mostrar porcentajes exactos para cada categoría
- [ ] Agregar celda markdown explicando qué son proporciones y su importancia
- [ ] Documentar proporciones de la variable objetivo (TenYearCHD) para identificar desbalanceo
- [ ] Incluir interpretación de las proporciones encontradas

### US-003: Respuesta a Preguntas del Dataset
**Descripción:** Como estudiante, necesito responder de forma clara y completa todas las preguntas planteadas sobre el dataset.

**Acceptance Criteria:**
- [ ] Crear sección "# 8. Respuestas a Preguntas del Dataset"
- [ ] Responder: ¿Cuántos registros hay? (con código y respuesta explícita)
- [ ] Responder: ¿Qué tipo de variable es cada una? (categórica, numérica continua, binaria, discreta)
- [ ] Responder: ¿Hay valores faltantes? (indicar cuáles y cuántos)
- [ ] Responder: ¿Hay valores fuera del rango esperado? (validar rangos médicos)
- [ ] Responder: ¿En qué tipo de dato están almacenados? (int64, float64, etc.)
- [ ] Responder: ¿Todas las variables son médicas o hay otra información? (clasificar por tipo)
- [ ] Responder: ¿Hay correlación entre las variables? (referenciar análisis US-001)
- [ ] Responder: ¿En qué rango está cada una? (min, max por variable)
- [ ] Responder: ¿Cómo son las proporciones de categóricas? (referenciar análisis US-002)
- [ ] Cada respuesta debe incluir código ejecutable y explicación en markdown

### US-004: Tratamiento de Outliers con Transformaciones
**Descripción:** Como científico de datos, necesito aplicar transformaciones a variables con outliers para reducir su impacto sin eliminar datos.

**Acceptance Criteria:**
- [ ] Crear sección "# 9. Tratamiento de Outliers"
- [ ] Agregar markdown explicando estrategias: log transform, winsorización, Box-Cox
- [ ] Para cada variable con outliers (totChol, sysBP, diaBP, BMI, heartRate, glucose):
  - [ ] Evaluar qué transformación es apropiada (evitar log si hay valores ≤0)
  - [ ] Aplicar transformación y crear nueva columna (ej: glucose_transformed)
  - [ ] Visualizar distribución antes/después con histogramas lado a lado
  - [ ] Documentar el efecto de la transformación
- [ ] Crear dataframe transformado para usar en el modelo
- [ ] Explicar en markdown por qué se eligió cada transformación

### US-005: Preparación de Datos para Modelado
**Descripción:** Como ML engineer, necesito preparar el dataset para entrenar el modelo de regresión logística.

**Acceptance Criteria:**
- [ ] Crear sección "# 10. Preparación para Modelado"
- [ ] Agregar markdown explicando la preparación de datos
- [ ] Verificar que no quedan valores nulos (o manejarlos)
- [ ] Definir X (features) e y (TenYearCHD)
- [ ] Decidir si usar variables originales o transformadas
- [ ] Aplicar train_test_split (80/20 o 70/30)
- [ ] Aplicar escalado/normalización si es necesario (StandardScaler o similar)
- [ ] Documentar decisiones tomadas en la preparación

### US-006: Entrenamiento del Modelo de Regresión Logística
**Descripción:** Como ML engineer, necesito entrenar un modelo de regresión logística para predecir TenYearCHD.

**Acceptance Criteria:**
- [ ] Crear sección "# 11. Modelo de Regresión Logística"
- [ ] Agregar markdown explicando qué es regresión logística
- [ ] Importar LogisticRegression de sklearn
- [ ] Entrenar modelo con datos de entrenamiento
- [ ] Generar predicciones en conjunto de test
- [ ] Documentar hiperparámetros utilizados (si hay ajustes)
- [ ] Código debe estar comentado y ser reproducible

### US-007: Evaluación del Modelo con Métricas Completas
**Descripción:** Como científico de datos, necesito evaluar el desempeño del modelo usando F1-Score, matriz de confusión y métricas adicionales.

**Acceptance Criteria:**
- [ ] Crear sección "## 11.1 Evaluación del Modelo"
- [ ] Agregar markdown explicando qué son precision, recall, F1-Score, accuracy
- [ ] Calcular y mostrar:
  - [ ] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-Score
  - [ ] Classification report completo
- [ ] Crear visualización de matriz de confusión con heatmap
- [ ] Agregar interpretación de cada métrica en el contexto médico
- [ ] Calcular y mostrar AUC-ROC si es posible
- [ ] Documentar si el modelo es útil para el problema planteado

### US-008: Análisis de Feature Importance
**Descripción:** Como científico de datos, necesito analizar qué variables son más importantes para la predicción y cómo contribuyen al modelo.

**Acceptance Criteria:**
- [ ] Crear sección "## 11.2 Análisis de Feature Importance"
- [ ] Agregar markdown explicando qué son los coeficientes en regresión logística
- [ ] Extraer coeficientes del modelo entrenado
- [ ] Crear DataFrame con features y sus coeficientes
- [ ] Ordenar por importancia (valor absoluto del coeficiente)
- [ ] Crear visualización de barras horizontales con los coeficientes
- [ ] Interpretar coeficientes: signo (+/-) y magnitud
- [ ] Identificar top 5 variables más influyentes
- [ ] Explicar qué significa cada coeficiente en términos médicos/prácticos
- [ ] Documentar si los resultados tienen sentido desde perspectiva médica

### US-009: Conclusiones Finales del Proyecto
**Descripción:** Como estudiante, necesito documentar conclusiones completas sobre el análisis realizado y el modelo desarrollado.

**Acceptance Criteria:**
- [ ] Actualizar sección "# 12. Conclusiones Finales" (o renumerar desde la actual #7)
- [ ] Resumir hallazgos principales del EDA
- [ ] Resumir respuestas a las preguntas del dataset
- [ ] Resumir hallazgos de correlaciones
- [ ] Resumir proporciones de variables categóricas
- [ ] Resumir tratamiento de outliers aplicado
- [ ] Resumir desempeño del modelo (métricas principales)
- [ ] Resumir variables más importantes para la predicción
- [ ] Incluir limitaciones del análisis y modelo
- [ ] Sugerir mejoras futuras o siguientes pasos

## Requisitos Funcionales

- **FR-1:** El notebook debe mantener el estilo actual con celdas markdown explicativas antes de cada sección de código
- **FR-2:** Cada función o técnica nueva debe incluir explicación teórica en markdown (qué es, para qué sirve, cómo interpretarla)
- **FR-3:** Todas las visualizaciones deben tener títulos, etiquetas y leyendas claras
- **FR-4:** El análisis de correlación debe incluir matriz completa y heatmap visual
- **FR-5:** Las proporciones de categóricas deben mostrarse en porcentajes y visualizarse con gráficos de barras
- **FR-6:** Las respuestas a preguntas deben ser explícitas y autocontenidas (no solo código)
- **FR-7:** El tratamiento de outliers debe aplicar transformaciones (log, winsorización) sin eliminar registros
- **FR-8:** El modelo de regresión logística debe usar sklearn.linear_model.LogisticRegression
- **FR-9:** La evaluación debe incluir: accuracy, precision, recall, F1-score, classification_report, matriz de confusión
- **FR-10:** El análisis de feature importance debe mostrar coeficientes del modelo ordenados por importancia
- **FR-11:** Cada coeficiente debe ser interpretado en contexto médico/práctico
- **FR-12:** Las conclusiones finales deben integrar todos los hallazgos del proyecto

## No Incluido (Out of Scope)

- No se aplicarán técnicas de balanceo de clases (SMOTE, oversampling, undersampling)
- No se crearán visualizaciones interactivas (solo matplotlib/seaborn estático)
- No se implementarán modelos alternativos (Random Forest, XGBoost, etc.)
- No se realizará validación cruzada extensiva (solo train/test split)
- No se implementará grid search para optimización de hiperparámetros
- No se crearán dashboards o aplicaciones interactivas
- No se realizará feature engineering avanzado (interacciones, polinomios)
- No se incluirá análisis temporal o de series de tiempo

## Consideraciones Técnicas

- **Notebook existente:** `resolucionTP.ipynb` ya tiene secciones 1-7 completas, agregar secciones 8-12
- **Librerías ya importadas:** numpy, pandas, matplotlib, seaborn, sklearn básico
- **Librerías adicionales necesarias:**
  - `from sklearn.linear_model import LogisticRegression`
  - `from sklearn.preprocessing import StandardScaler` (si se requiere escalado)
  - `from sklearn.metrics import roc_auc_score, roc_curve` (para AUC-ROC opcional)
  - `from scipy.stats import boxcox` (si se usa Box-Cox transform)
- **Dataset:** `framingham.csv` ya cargado en DataFrame `df`
- **Variables tratadas:** Valores nulos ya imputados en secciones anteriores
- **Estilo de código:** Funciones reutilizables con docstrings cuando sea apropiado
- **Visualizaciones:** Usar `figsize=(14, 10)` o similar para gráficos grandes
- **Formato markdown:** Usar encabezados jerárquicos (#, ##, ###), negritas, listas

## Métricas de Éxito

- Todas las 9 preguntas sobre el dataset respondidas con código y explicación clara
- Matriz de correlación visualizada e interpretada correctamente
- Proporciones de todas las variables categóricas calculadas y visualizadas
- Transformaciones aplicadas a variables con outliers y documentadas
- Modelo de regresión logística entrenado con F1-Score ≥ 0.50 (ajustar expectativa según datos)
- Matriz de confusión visualizada y métricas completas calculadas
- Top 5 features más importantes identificadas e interpretadas
- Notebook ejecutable de inicio a fin sin errores
- Todas las celdas con output visible (no vacías)
- Conclusiones completas y coherentes con el análisis realizado

## Preguntas Abiertas

- ¿Se debe guardar el modelo entrenado en un archivo (pickle/joblib)?
- ¿Se requiere un archivo PDF exportado del notebook completo?
- ¿Hay un formato específico para la presentación de resultados?
- ¿Se necesita un resumen ejecutivo separado del notebook?

## Notas de Implementación

### Orden de Implementación Sugerido:
1. US-003 (Respuestas a preguntas) - aprovechar análisis ya realizado
2. US-001 (Correlaciones) - análisis estadístico
3. US-002 (Proporciones) - análisis de categóricas
4. US-004 (Tratamiento outliers) - preparación de datos
5. US-005 (Preparación modelado) - split y escalado
6. US-006 (Entrenamiento) - crear modelo
7. US-007 (Evaluación) - métricas
8. US-008 (Feature importance) - interpretación
9. US-009 (Conclusiones) - síntesis final

### Transformaciones Recomendadas por Variable:
- **glucose:** Tiene valores con asimetría → Log transform o Box-Cox
- **totChol, sysBP, diaBP:** Outliers moderados → Winsorización (percentiles 1-99)
- **BMI:** Distribución razonablemente normal → Winsorización ligera
- **heartRate:** Outliers leves → Puede no requerir transformación o winsorización suave

### Consideraciones de Escalado:
Regresión logística se beneficia de escalado cuando las variables tienen magnitudes muy diferentes. Evaluar:
- sysBP, diaBP ~100-200
- age ~30-70
- glucose ~50-400
- → **Aplicar StandardScaler** antes del modelado

---

**Fecha de creación:** 2026-01-16
**Versión:** 1.0
**Autor:** PRD Generator Skill
