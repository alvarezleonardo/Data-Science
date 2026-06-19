# Ejercicio 7: Predicción de Lluvia con XGBoost

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de Machine Learning para predecir si lloverá mañana basándose en datos meteorológicos históricos de Australia. Se comparan dos algoritmos de clasificación: Árbol de Decisión tradicional y XGBoost (Extreme Gradient Boosting).

## 🎯 Objetivos

- Realizar preprocesamiento de datos meteorológicos
- Implementar un modelo baseline con Árbol de Decisión
- Desarrollar un modelo avanzado con XGBoost
- Comparar el rendimiento de ambos modelos
- Evaluar métricas de clasificación

## 📊 Dataset

**Fuente:** Weather Australia Dataset (`weatherAUS.csv.zip`)

**Descripción:** Contiene información meteorológica diaria de diversas estaciones en Australia, incluyendo:
- Temperaturas máximas y mínimas
- Humedad
- Presión atmosférica
- Velocidad y dirección del viento
- Precipitaciones
- Variable objetivo: `RainTomorrow` (¿Lloverá mañana?)

## 🔧 Tecnologías y Librerías

```python
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
```

## 📁 Estructura del Proyecto

```
Ejercicio 7 XGBoost/
├── Ejercicio 7.ipynb       # Notebook principal con análisis completo
├── weatherAUS.csv.zip      # Dataset de datos meteorológicos
└── README.md               # Este archivo
```

## 🚀 Instalación y Ejecución

### Requisitos Previos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Ejecución

1. Clona o descarga este repositorio
2. Asegúrate de que el archivo `weatherAUS.csv.zip` esté en el mismo directorio
3. Abre `Ejercicio 7.ipynb` en Jupyter Notebook o JupyterLab
4. Ejecuta las celdas secuencialmente

## 📖 Metodología

### 1. **Preprocesamiento de Datos**
   - Eliminación de columnas con muchos valores faltantes
   - Eliminación de filas con valores nulos
   - Eliminación de variables categóricas no utilizadas
   - Eliminación de variables correlacionadas para evitar multicolinealidad
   - Codificación de la variable objetivo (Yes/No → 1/0)

### 2. **Selección de Features**
   Se seleccionaron dos variables numéricas clave:
   - `MaxTemp`: Temperatura máxima del día
   - `Humidity3pm`: Humedad a las 3 PM

### 3. **División de Datos**
   - 70% para entrenamiento
   - 30% para prueba
   - Estratificación por clase para mantener la proporción

### 4. **Modelado**

   #### Modelo 1: Árbol de Decisión (Baseline)
   - Algoritmo: `DecisionTreeClassifier`
   - Configuración: Parámetros por defecto
   - Propósito: Establecer una línea base de rendimiento

   #### Modelo 2: XGBoost
   - Algoritmo: `XGBClassifier`
   - Configuración: `eval_metric='logloss'`
   - Propósito: Mejorar el rendimiento mediante ensemble boosting

### 5. **Evaluación**
   Se utilizan tres métricas principales:
   - **Accuracy**: Exactitud general del modelo
   - **Classification Report**: Precision, Recall, F1-Score por clase
   - **Confusion Matrix**: Distribución de predicciones correctas e incorrectas

## 📊 Métricas de Evaluación

### Accuracy (Exactitud)
Proporción de predicciones correctas sobre el total:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision (Precisión)
De las predicciones positivas, cuántas fueron correctas:
```
Precision = TP / (TP + FP)
```

### Recall (Sensibilidad)
De los casos positivos reales, cuántos fueron detectados:
```
Recall = TP / (TP + FN)
```

### F1-Score
Media armónica entre Precision y Recall:
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Matriz de Confusión
| | Predicción: No | Predicción: Yes |
|---|---|---|
| **Real: No** | TN (Verdaderos Negativos) | FP (Falsos Positivos) |
| **Real: Yes** | FN (Falsos Negativos) | TP (Verdaderos Positivos) |

## 🔍 ¿Qué es XGBoost?

**XGBoost (Extreme Gradient Boosting)** es un algoritmo de ensemble learning que:

- Combina múltiples árboles de decisión débiles en un modelo fuerte
- Utiliza gradient boosting: cada árbol nuevo corrige los errores de los anteriores
- Implementa regularización para prevenir overfitting
- Es altamente eficiente y escalable
- Ha ganado numerosas competencias de Kaggle

**Ventajas sobre árboles individuales:**
- Mayor precisión y generalización
- Mejor manejo de datos desbalanceados
- Robusto ante outliers
- Capacidad de capturar relaciones complejas

## 📈 Resultados Esperados

El notebook muestra:
- Comparación directa entre Árbol de Decisión y XGBoost
- Análisis detallado de las métricas de cada modelo
- Visualización de matrices de confusión
- Conclusiones sobre qué modelo tiene mejor rendimiento

## 🔮 Próximos Pasos y Mejoras

1. **Optimización de Hiperparámetros:**
   - Ajustar `learning_rate`, `max_depth`, `n_estimators`
   - Utilizar Grid Search o Random Search
   - Implementar validación cruzada

2. **Ingeniería de Features:**
   - Incorporar más variables meteorológicas
   - Crear features de interacción
   - Agregar variables temporales (día de la semana, estación)

3. **Manejo de Desbalanceo:**
   - Aplicar técnicas de oversampling (SMOTE)
   - Ajustar pesos de clases
   - Probar métricas alternativas (ROC-AUC)

4. **Análisis Adicional:**
   - Feature importance (importancia de variables)
   - Curvas de aprendizaje
   - Análisis de errores

5. **Otros Modelos:**
   - Random Forest
   - LightGBM
   - CatBoost
   - Redes Neuronales

## 👨‍💻 Autor

Leo Alvarez - Digital House Data Science

## 📝 Notas

- Este proyecto es parte de un ejercicio académico
- Los datos son públicos y pueden contener valores atípicos
- Se recomienda ejecutar el notebook completo para reproducir los resultados
- La semilla aleatoria (`random_state=42`) asegura reproducibilidad

## 📚 Referencias

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Weather Australia Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

---

**Fecha:** Enero 2026  
**Curso:** Data Science - Digital House
