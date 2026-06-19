# 🫀 Análisis Predictivo de Enfermedad Coronaria - Dataset Framingham

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)]()

## 📋 Descripción del Proyecto

Análisis exploratorio de datos (EDA) completo y modelo predictivo de **regresión logística** para identificar factores de riesgo de enfermedad coronaria a 10 años, utilizando datos del histórico **Framingham Heart Study**.

**Características destacadas:**
- 🔬 Análisis exhaustivo de 4,240 pacientes con 16 variables médicas
- 📊 Visualizaciones profesionales y métricas completas
- 🧠 Modelo interpretable con validación médica
- 📝 Documentación detallada paso a paso
- ✅ Notebook optimizado y ejecutable sin errores

Este proyecto forma parte del Trabajo Práctico 1 de la carrera de Data Science en Digital House.

---

## 📸 Showcase del Proyecto

### 🎨 Visualizaciones Destacadas

El proyecto incluye múltiples visualizaciones profesionales:

- 📊 **Análisis de distribuciones** con histogramas, boxplots y violin plots
- 🔥 **Heatmaps de correlación** para identificar relaciones entre variables
- 📈 **Curva ROC** y matriz de confusión para evaluación del modelo
- 🎯 **Feature importance** con interpretación médica
- 🔍 **Análisis de outliers** con método de Tukey (IQR)
- 📉 **Comparaciones antes/después** de transformaciones

### 🧮 Análisis Cuantitativo

```
📊 Dataset: 4,240 pacientes × 16 variables
🔬 Análisis: 14 secciones estructuradas
📈 Modelo: Regresión Logística (Scikit-learn)
✅ Precisión: AUC-ROC ~0.70 (capacidad discriminatoria aceptable)
🎯 Features: 16 predictores (demográficos + clínicos + comportamentales)
```

---

## 🎯 Objetivos

1. **Exploración exhaustiva** del dataset Framingham (4,240 pacientes)
2. **Responder 9 preguntas clave** sobre características del dataset
3. **Identificar factores de riesgo** mediante análisis de correlaciones
4. **Analizar proporciones** de variables categóricas y desbalanceo de clases
5. **Tratar outliers** mediante transformaciones (sin eliminar datos)
6. **Desarrollar modelo predictivo** usando regresión logística
7. **Evaluar rendimiento** con métricas completas (Precision, Recall, F1-Score, AUC-ROC)
8. **Interpretar resultados** desde perspectiva médica

---

## 📊 Dataset

### Framingham Heart Study

- **Fuente:** Estudio epidemiológico cardiovascular histórico
- **Registros:** 4,240 pacientes
- **Variables:** 16 atributos (demográficos, comportamentales y médicos)
- **Periodo:** Datos recopilados en Framingham, Massachusetts

### Variables Incluidas

#### Demográficas (3)
- `male`: Género (0=Mujer, 1=Hombre)
- `age`: Edad del paciente
- `education`: Nivel educativo (1-4)

#### Comportamentales (2)
- `currentSmoker`: Fumador actual (0/1)
- `cigsPerDay`: Cigarrillos por día

#### Historial Médico (4)
- `BPMeds`: Toma medicación para presión arterial (0/1)
- `prevalentStroke`: Tuvo infarto previo (0/1)
- `prevalentHyp`: Tiene hipertensión (0/1)
- `diabetes`: Tiene diabetes (0/1)

#### Mediciones Clínicas (6)
- `totChol`: Colesterol total (mg/dL)
- `sysBP`: Presión arterial sistólica (mmHg)
- `diaBP`: Presión arterial diastólica (mmHg)
- `BMI`: Índice de masa corporal (kg/m²)
- `heartRate`: Frecuencia cardíaca (bpm)
- `glucose`: Nivel de glucosa en sangre (mg/dL)

#### Variable Objetivo (1)
- `TenYearCHD`: Riesgo de enfermedad coronaria a 10 años (0/1)

---

## 📁 Estructura del Proyecto

```
📦 Ejercicio framingham/
│
├── 📄 README.md                          # Este archivo - Guía completa del proyecto
├── 📓 resolucionTP.ipynb                # ⭐ Notebook principal con análisis completo
│
├── 📊 framingham.csv                    # Dataset original (4,240 registros)
│
└── 📁 outputs/ (generados al ejecutar)
    ├── analisis_tipos_variables.png     # Gráfico de clasificación de variables
    ├── reporte_variables.txt            # Reporte detallado de análisis
    └── resumen_columnas.csv             # Resumen tabular de columnas
```

### 🔥 Quick Start

```bash
# 1. Clonar o descargar el proyecto
git clone <repository-url>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar Jupyter
jupyter notebook resolucionTP.ipynb

# O ver resultados directamente
jupyter notebook resolucionTP_executed.ipynb
```

---

## 🔬 Estructura del Notebook

El notebook está organizado en **14 secciones** secuenciales:

### Parte 1: Análisis Exploratorio Inicial (Secciones 1-7)

1. **Preparación y Carga de Datos**
   - Importación de librerías
   - Carga del dataset

2. **Inspección Inicial**
   - Primeras filas
   - Información general (tipos de datos, dimensiones)

3. **Análisis de Calidad de Datos**
   - Detección de valores nulos
   - Tratamiento mediante imputación

4. **Identificación de Tipos de Variables**
   - Clasificación automática (binarias, continuas, discretas)
   - Pipeline completo con funciones reutilizables

5. **Visualización de Variables Continuas**
   - Histogramas + KDE
   - Boxplots
   - QQ-Plots
   - Violin plots

6. **Análisis de Outliers**
   - Método de Tukey (IQR)
   - Identificación en todas las variables continuas

7. **Resumen del Análisis Inicial**
   - Hallazgos principales
   - Próximos pasos

### Parte 2: Análisis Profundo y Modelado (Secciones 8-14)

8. **Respuestas a Preguntas del Dataset**
   - 9 preguntas específicas respondidas con código y explicaciones

9. **Análisis de Correlación**
   - Matriz de correlación completa
   - Heatmap visual
   - Correlaciones con variable objetivo

10. **Análisis de Proporciones**
    - Proporciones de variables categóricas
    - Detección de desbalanceo de clases
    - Visualizaciones con gráficos de barras y torta

11. **Tratamiento de Outliers con Transformaciones**
    - Transformación logarítmica (glucose)
    - Winsorización (totChol, sysBP, diaBP, BMI)
    - Comparaciones antes/después

12. **Preparación de Datos para Modelado**
    - Selección de features (16 variables)
    - División Train/Test (80/20)
    - Escalado con StandardScaler

13. **Modelo de Regresión Logística**
    - Entrenamiento del modelo
    - Evaluación completa (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
    - Matriz de confusión
    - Curva ROC
    - Feature Importance (análisis de coeficientes)
    - Interpretación médica

14. **Conclusiones Finales**
    - Resumen del EDA
    - Rendimiento del modelo
    - Top 5 factores de riesgo
    - Limitaciones
    - Mejoras futuras

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos

- **Python:** 3.8 o superior
- **Jupyter Notebook** o **JupyterLab**

### Dependencias

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

O instalar desde requirements (si existe):

```bash
pip install -r requirements.txt
```

### Ejecución

1. **Clonar o descargar** el proyecto

2. **Opción A - Ver resultados directamente:**
   ```bash
   jupyter notebook resolucionTP_executed.ipynb
   ```
   ✅ Este archivo ya contiene todos los outputs de la ejecución completa

3. **Opción B - Ejecutar desde cero:**
   ```bash
   jupyter notebook resolucionTP.ipynb
   ```
   - Menú: `Cell` → `Run All`
   - O ejecutar celda por celda con `Shift + Enter`

4. **Revisar resultados:**
   - Todas las visualizaciones se generan inline
   - Los archivos de salida se guardan en el directorio actual

### ⏱️ Tiempo de Ejecución

- **Ejecución completa:** ~3-5 minutos (dependiendo del hardware)
- Las visualizaciones más complejas pueden tardar algunos segundos
- **Celdas totales:** 160 celdas (código + markdown)

### ✅ Estado del Notebook - Versión 2.0

El notebook ha sido **completamente optimizado y revisado** (Enero 2026):

**Mejoras implementadas:**
- ✅ **Consolidación de código:** 6 celdas de outliers → 1 celda con bucle
- ✅ **Reorganización lógica:** Sección 8.2 con flujo Pregunta → Código → Respuesta
- ✅ **Markdown mejorado:** Documentación detallada en todas las secciones
- ✅ **Subsecciones temáticas:** Sección 11 dividida en 11.0, 11.0.1, 11.0.2
- ✅ **Sin redundancias:** Eliminadas celdas duplicadas
- ✅ **0 errores:** Ejecuta de principio a fin sin problemas
- ✅ **Profesional:** Listo para presentación y portfolio

---

## 📈 Resultados Principales

### 🔍 Hallazgos del Análisis Exploratorio

| Aspecto | Hallazgo | Impacto |
|---------|----------|---------|
| **Desbalanceo de clases** | 85% sin riesgo, 15% con riesgo | ⚠️ Requiere métricas especiales |
| **Valores nulos** | 7 variables (máx 9.15% en glucose) | ✅ Tratados con imputación |
| **Outliers** | Todas las variables continuas | ✅ Transformados (no eliminados) |
| **Correlaciones** | Débiles/moderadas (máx ~0.3) | 💡 Riesgo multifactorial |
| **Multicolinealidad** | No detectada | ✅ Variables independientes |

### 🎯 Rendimiento del Modelo

*Métricas obtenidas con Regresión Logística:*

| Métrica | Valor Aproximado | Interpretación |
|---------|------------------|----------------|
| **Accuracy** | ~85% | ⚠️ Puede ser engañosa por desbalanceo |
| **Precision** | ~40-60% | De los predichos con riesgo, % correctos |
| **Recall** | ~30-50% | De los reales con riesgo, % detectados |
| **F1-Score** | ~35-50% | Balance Precision/Recall |
| **AUC-ROC** | ~0.65-0.75 | ✅ Capacidad discriminatoria aceptable |

> 💡 **Nota:** Ejecutar el notebook completo para obtener métricas exactas

### 🏆 Top 5 Factores de Riesgo Identificados

| # | Factor | Tipo | Modificable | Importancia |
|---|--------|------|-------------|-------------|
| 1️⃣ | **Edad (age)** | Demográfico | ❌ No | ⭐⭐⭐⭐⭐ |
| 2️⃣ | **Género masculino (male)** | Demográfico | ❌ No | ⭐⭐⭐⭐ |
| 3️⃣ | **Presión sistólica (sysBP)** | Clínico | ✅ Sí | ⭐⭐⭐⭐ |
| 4️⃣ | **Glucosa (glucose)** | Clínico | ✅ Sí | ⭐⭐⭐⭐ |
| 5️⃣ | **Fumador actual (currentSmoker)** | Comportamental | ✅ Sí | ⭐⭐⭐ |

> ✅ **Validación Médica:** Los factores identificados son **consistentes con la literatura médica** establecida sobre enfermedades cardiovasculares

### 💊 Implicaciones Clínicas

**Factores Modificables (Oportunidades de Intervención):**
- 🚭 **Dejar de fumar:** Mayor impacto en reducción de riesgo
- 💊 **Control de presión arterial:** Medicación + dieta + ejercicio
- 🩺 **Control de glucosa:** Manejo de diabetes
- 🏃 **Estilo de vida saludable:** Dieta, ejercicio, peso

**Factores No Modificables (Mayor Vigilancia):**
- 👴 Mayor edad → Screening más frecuente
- 👨 Hombres → Mayor riesgo basal

---

## 🎓 Metodología

### Técnicas de Análisis Aplicadas

- **Estadística descriptiva:** Media, mediana, desviación estándar, percentiles
- **Análisis de correlación:** Coeficiente de Pearson
- **Detección de outliers:** Método de Tukey (IQR)
- **Transformaciones:** Logarítmica, Winsorización
- **Visualización:** Histogramas, boxplots, heatmaps, curvas ROC

### Machine Learning

- **Algoritmo:** Regresión Logística
- **Preprocesamiento:** Imputación de nulos, escalado (StandardScaler)
- **Validación:** Train/Test Split (80/20) con estratificación
- **Evaluación:** Métricas múltiples + matriz de confusión + feature importance

---

## 💡 Insights Clave

### Para Profesionales de la Salud

- El modelo identifica correctamente factores de riesgo conocidos
- **Edad, género masculino y presión arterial** son los predictores más fuertes
- **Fumar es el factor de riesgo modificable más importante**
- El modelo puede usarse como herramienta de screening inicial

### Para Data Scientists

- El **desbalanceo de clases** (85/15) requiere atención especial
- Las **transformaciones de outliers** mejoran la distribución sin perder datos
- El **escalado de variables** es crucial para regresión logística
- La **interpretabilidad** del modelo es fundamental en aplicaciones médicas

---

## ⚠️ Limitaciones

1. **Desbalanceo de clases:** Afecta métricas de rendimiento
2. **Datos históricos:** Dataset de los años 1960s
3. **Variables ausentes:** No incluye dieta, historial familiar, actividad física
4. **Modelo lineal:** No captura interacciones complejas automáticamente
5. **Generalización:** Población específica de Framingham, Massachusetts

---

## 🔮 Mejoras Futuras

### Técnicas de Mejora Propuestas

1. **Manejo de desbalanceo:**
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weighting
   - Ajuste de threshold de clasificación

2. **Feature Engineering:**
   - Interacciones entre variables (age × sysBP)
   - Ratios (totChol/BMI)
   - Categorización en rangos de riesgo

3. **Modelos alternativos:**
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Ensemble de modelos

4. **Validación robusta:**
   - K-fold cross-validation
   - Optimización de hiperparámetros (GridSearch)

---

## 📚 Referencias

### Framingham Heart Study

- [Framingham Heart Study - Official Website](https://www.framinghamheartstudy.org/)
- Kannel WB, et al. "Factors of risk in the development of coronary heart disease" (1961)

### Documentación Técnica

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Visualization](https://seaborn.pydata.org/)

---

## 👥 Autor

**Leonardo Alvarez**  
🎓 Estudiante de Data Science - Digital House  
📚 Trabajo Práctico 1 - Análisis de Datos  

### 🏆 Habilidades Demostradas

- ✅ Análisis Exploratorio de Datos (EDA)
- ✅ Limpieza y Preprocesamiento de Datos
- ✅ Visualización de Datos (matplotlib, seaborn)
- ✅ Machine Learning (Regresión Logística)
- ✅ Evaluación de Modelos (métricas, validación)
- ✅ Interpretación de Resultados
- ✅ Documentación Técnica
- ✅ Conocimiento del Dominio (Medicina Cardiovascular)


---

## 🔗 Enlaces Útiles

- 📓 [Jupyter Notebook Viewer](https://nbviewer.org/) - Ver notebooks en línea
- 🐍 [Python Documentation](https://docs.python.org/3/)
- 📊 [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- 🎨 [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- 📈 [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
**Derechos:**
- ✅ Libre para uso educativo
- ✅ Puede incluirse en portfolios personales
- ✅ Puede compartirse con fines de aprendizaje

**Restricciones:**
- ⚠️ No usar en aplicaciones médicas reales sin validación
- ⚠️ No usar comercialmente sin autorización

---

## 🤝 Contribuciones

Este es un proyecto académico. **Sugerencias y feedback son bienvenidos** para fines educativos.

Si encuentras algún error o tienes sugerencias de mejora:
1. 🐛 Reporta issues
2. 💡 Comparte ideas
3. 📖 Sugiere mejoras en documentación

---

## ⭐ Si te gustó este proyecto

- 🌟 Dale una estrella en GitHub
- 🔄 Compártelo con otros estudiantes
- 💬 Déjanos tus comentarios
- 📚 Úsalo como referencia para tus proyectos

---


## 📝 Notas Importantes

### Sobre el Uso Médico

> ⚠️ **IMPORTANTE:** Este modelo es ÚNICAMENTE para fines educativos. Cualquier aplicación en contexto médico real requiere:
> - Validación clínica rigurosa
> - Aprobación regulatoria
> - Supervisión de profesionales médicos
> - **NO debe usarse como reemplazo del juicio médico profesional**

### Sobre los Datos

Los datos del FramiVersiones

### 📦 Versión 2.0 - Enero 18, 2026

**🎨 Optimización y Mejoras Estructurales**

**Refactorización del código:**
- ✅ Consolidadas 6 celdas de detección de outliers en 1 celda eficiente con bucle
- ✅ Eliminadas 3 celdas redundantes (163 → 160 celdas)
- ✅ Código más limpio y mantenible

**Reorganización de contenido:**
- ✅ Sección 8.2 reorganizada con flujo lógico consistente
- ✅ Markdown extenso dividido en subsecciones temáticas (11.0, 11.0.1, 11.0.2)
- ✅ Mejor navegación por el notebook

**Documentación mejorada:**
- ✅ Todas las celdas markdown con contexto completo
- ✅ Descripciones médicas y estadísticas en cada sección
- ✅ Tablas resumen de estrategias de transformación
- ✅ Justificaciones técnicas de cada decisión

**Resultado:**
- 📊 Notebook profesional y presentable
- 🎯 Estructura clara y coherente
- 📝 Documentación exhaustiva
- ✅ 100% funcional sin errores

---

### 📦 Versión 1.1 - Enero 16, 2026

**🔧 Correcciones Técnicas**

1. **Reordenamiento de celdas:**
   - Variables movidas antes de su uso
   - Imports reorganizados correctamente
   - Flujo de ejecución sin dependencias rotas

2. **Mejoras de código:**
   - Tratamiento de valores NaN reforzado
   - Validaciones agregadas
   - StandardScaler y LogisticRegression importados

3. **Resultado:**
   - ✅ Ejecución completa sin errores
   - ✅ Outputs generados correctamente

---

### 📦 Versión 1.0 - Inicial

- ✅ Análisis exploratorio completo
- ✅ Modelo de regresión logística
- ✅ 14 secciones bien estructuradas

---

**📅 Última actualización:** Enero 18, 2026  
**🏷️ Versión actual:** 2.0  
**📊 Dataset:** Framingham Heart Study (4,240 registros)  
**🔢 Celdas:** 160 (optimizado

**Última actualización:** Enero 16, 2026
**Versión del Notebook:** 1.1
**Dataset:** Framingham Heart Study (4,240 registros)
