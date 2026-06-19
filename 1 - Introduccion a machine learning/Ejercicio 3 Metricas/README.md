# Métricas de Evaluación de Modelos de Clasificación

## 📊 Descripción
Este proyecto implementa un modelo de **Regresión Logística** para predecir la rotación de empleados (employee turnover) utilizando datos de recursos humanos. El enfoque principal está en el análisis exhaustivo de métricas de evaluación de modelos de clasificación binaria.

## 📁 Dataset
**Archivo**: `HR_comma_sep.csv`

El dataset contiene información sobre empleados con las siguientes características:
- `satisfaction_level`: Nivel de satisfacción del empleado (0-1)
- `last_evaluation`: Última evaluación de desempeño (0-1)
- `number_project`: Número de proyectos asignados
- `average_montly_hours`: Promedio de horas mensuales trabajadas
- `time_spend_company`: Tiempo en la empresa (años)
- `Work_accident`: Si tuvo accidentes laborales (0 = No, 1 = Sí)
- `promotion_last_5years`: Si tuvo promoción en los últimos 5 años (0 = No, 1 = Sí)
- `left`: **Variable objetivo** (1 = dejó la empresa, 0 = permanece)

## 📂 Estructura del Proyecto
```
Ejercicio 3 Metricas/
├── Ejercicio3.ipynb       # Notebook principal con análisis completo
├── HR_comma_sep.csv       # Dataset de recursos humanos
└── README.md              # Documentación del proyecto
```

## 🔄 Workflow del Análisis

### 1. Preparación de Datos
- Importación de librerías (scikit-learn, pandas, numpy, matplotlib)
- Carga y exploración del dataset
- Selección de variables predictoras
- División train/test (67%/33%)
- Normalización con StandardScaler

### 2. Modelado
- Creación de modelo de Regresión Logística (C=1e10)
- Entrenamiento con datos normalizados
- Generación de predicciones

### 3. Evaluación Completa del Modelo

#### Métricas Básicas
- **Matriz de Confusión**: Visualización de TP, TN, FP, FN
- **Accuracy**: Porcentaje de predicciones correctas
- **Tasa de Error**: Complemento de la exactitud
- **Análisis de Balance de Clases**: Distribución del dataset

#### Métricas Avanzadas
- **Recall (Sensibilidad)**: Capacidad de detectar empleados en riesgo
- **Specificity (Especificidad)**: Capacidad de identificar empleados que permanecerán
- **F1 Score**: Balance entre Precision y Recall

#### Análisis de Probabilidades
- **Predicciones Probabilísticas**: Análisis de confianza del modelo
- **Histograma de Probabilidades**: Distribución de predicciones
- **Ajuste de Threshold**: Cambio de umbral de 0.5 a 0.3
- **Comparación de Resultados**: Impacto del threshold en las métricas

#### Análisis ROC
- **Curva ROC**: Evaluación del rendimiento en todos los thresholds
- **AUC (Area Under Curve)**: Métrica única de discriminación del modelo

## 🎯 Métricas Implementadas

| Métrica | Descripción | Uso Principal |
|---------|-------------|---------------|
| **Confusion Matrix** | Matriz de clasificación (TP, TN, FP, FN) | Base para todas las métricas |
| **Accuracy** | Predicciones correctas / Total | Datasets balanceados |
| **Recall** | TP / (TP + FN) | Minimizar falsos negativos |
| **Specificity** | TN / (TN + FP) | Minimizar falsos positivos |
| **F1 Score** | Media armónica Precision-Recall | Datasets desbalanceados |
| **ROC Curve** | TPR vs FPR en todos los thresholds | Comparación de modelos |
| **AUC** | Área bajo curva ROC | Poder discriminatorio general |

## 🛠️ Requisitos Técnicos

### Software
- Python 3.12.7+
- Jupyter Notebook / VS Code

### Librerías
```bash
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 📥 Instalación

```bash
# Clonar o descargar el proyecto
cd "Ejercicio 3 Metricas"

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install scikit-learn pandas numpy matplotlib
```

## 🚀 Uso

1. Abrir el notebook `Ejercicio3.ipynb` en Jupyter o VS Code
2. Verificar que `HR_comma_sep.csv` esté en el mismo directorio
3. Ejecutar las celdas secuencialmente
4. Cada sección incluye explicaciones detalladas de las métricas y su aplicación

## 📈 Resultados Clave

- El modelo logra una buena capacidad de discriminación entre empleados en riesgo y empleados estables
- El análisis de threshold demuestra el trade-off entre sensibilidad y especificidad
- La curva ROC y el AUC permiten evaluar el modelo independientemente del umbral de decisión
- Las explicaciones detalladas facilitan la comprensión de cuándo usar cada métrica

## 📚 Conceptos Aprendidos

- Interpretación de matriz de confusión
- Diferencia entre Accuracy, Precision, Recall y F1
- Importancia del balance de clases
- Ajuste de thresholds según objetivos de negocio
- Análisis ROC y AUC para evaluación robusta
- Trade-offs entre diferentes métricas

## 👤 Autor
**Leo Alvarez**

## 📅 Fecha
Diciembre 2025

---

*Proyecto educativo enfocado en el análisis de métricas de evaluación de modelos de Machine Learning*
