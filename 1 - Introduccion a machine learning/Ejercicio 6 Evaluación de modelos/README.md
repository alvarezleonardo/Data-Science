# Ejercicio 6: Evaluación de Modelos de Machine Learning

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de clasificación utilizando el algoritmo **K-Nearest Neighbors (KNN)** sobre el dataset de Abalone. El objetivo principal es comparar diferentes metodologías de optimización de hiperparámetros y evaluar el rendimiento de modelos de clasificación.

## 🎯 Objetivos

1. **Explorar y preparar** el dataset de Abalone para clasificación binaria
2. **Implementar validación cruzada** estratificada para evaluación robusta
3. **Comparar tres enfoques** de búsqueda de hiperparámetros:
   - Búsqueda manual con loop
   - GridSearchCV (búsqueda exhaustiva)
   - RandomizedSearchCV (búsqueda aleatoria)
4. **Evaluar métricas** de clasificación: accuracy, precision, recall, F1-score
5. **Analizar el trade-off** entre sesgo y varianza en modelos KNN

## 📊 Dataset

**Nombre**: Abalone Dataset
- **Archivo**: `abalone.csv`
- **Separador**: Punto y coma (`;`)
- **Variable objetivo**: `Adulto` (clasificación binaria)
- **Variables eliminadas**: `Sex` (categórica) y `Adulto` (target)
- **Características**: Medidas físicas de abalones (moluscos marinos)

## 🛠️ Tecnologías y Librerías

```python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
  - KNeighborsClassifier
  - train_test_split
  - StandardScaler
  - StratifiedKFold
  - cross_val_score
  - GridSearchCV
  - RandomizedSearchCV
  - confusion_matrix
  - classification_report
```

## 📁 Estructura del Proyecto

```
Ejercicio 6 Evaluación de modelos/
│
├── abalone.csv              # Dataset de Abalone
├── ejercicio6.ipynb         # Notebook principal con el análisis
└── README.md               # Este archivo
```

## 🔍 Metodología

### 1. Preprocesamiento de Datos
- **Carga**: Lectura del CSV con separador personalizado
- **Separación**: Features (X) y Target (y)
- **Análisis**: Distribución de clases para detectar desbalance
- **División**: 70% entrenamiento, 30% prueba
- **Estandarización**: StandardScaler para normalizar variables (media=0, std=1)

### 2. Búsqueda Manual de Hiperparámetros
- **Rango de K**: 1 a 30 vecinos
- **Validación Cruzada**: StratifiedKFold con 10 folds
- **Métrica**: Accuracy promedio
- **Visualización**: Gráfico de K vs Accuracy

### 3. GridSearchCV - Búsqueda Exhaustiva
- **Parámetros**: n_neighbors de 1 a 30
- **Validación**: 10-fold stratified cross-validation
- **Ventajas**: Prueba todas las combinaciones posibles
- **Desventajas**: Computacionalmente costoso
- **Total de ajustes**: 30 valores × 10 folds = 300 entrenamientos

### 4. RandomizedSearchCV - Búsqueda Aleatoria
- **Parámetros**: Mismo rango que GridSearchCV
- **Iteraciones**: 20 combinaciones aleatorias
- **Ventajas**: Más rápido, eficiente para grandes espacios de búsqueda
- **Desventajas**: Puede no encontrar el óptimo global
- **Total de ajustes**: 20 valores × 10 folds = 200 entrenamientos

## 📈 Resultados y Evaluación

### Métricas Utilizadas

1. **Accuracy**: Precisión global del modelo
2. **Precision**: TP / (TP + FP) - Exactitud de predicciones positivas
3. **Recall**: TP / (TP + FN) - Capacidad de detectar positivos reales
4. **F1-Score**: Media armónica de precision y recall
5. **Matriz de Confusión**: Visualización de TP, TN, FP, FN

### Interpretación de la Matriz de Confusión

```
                Predicho Negativo    Predicho Positivo
Real Negativo       TN                    FP
Real Positivo       FN                    TP
```

- **TN (True Negative)**: Correctamente clasificados como negativos
- **TP (True Positive)**: Correctamente clasificados como positivos
- **FP (False Positive)**: Error Tipo I - Falsa alarma
- **FN (False Negative)**: Error Tipo II - No detectado

## 🔑 Conceptos Clave

### K-Nearest Neighbors (KNN)
- Algoritmo de clasificación basado en **similitud por distancia**
- **K pequeño**: Más sensible al ruido (alto variance, bajo bias)
- **K grande**: Más generalizado (bajo variance, alto bias)
- Requiere **estandarización** de variables

### Validación Cruzada Estratificada
- Divide datos en K folds manteniendo **proporción de clases**
- Reduce **variance** en la estimación del rendimiento
- Utiliza todos los datos para **entrenamiento y validación**

### Trade-off Sesgo-Varianza
- **Sesgo (Bias)**: Error por suposiciones erróneas
- **Varianza (Variance)**: Error por sensibilidad a fluctuaciones
- **Objetivo**: Encontrar el balance óptimo

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecución
1. Abrir `ejercicio6.ipynb` en Jupyter Notebook o VS Code
2. Asegurarse de que `abalone.csv` esté en el mismo directorio
3. Ejecutar las celdas secuencialmente
4. Analizar los resultados y visualizaciones

## 💡 Conclusiones

1. **Búsqueda Manual**: 
   - Útil para entender el comportamiento del hiperparámetro
   - Permite visualización detallada de la relación K vs Accuracy

2. **GridSearchCV**:
   - Garantiza encontrar el mejor K dentro del rango especificado
   - Recomendado cuando el espacio de búsqueda es pequeño
   - Más lento pero más exhaustivo

3. **RandomizedSearchCV**:
   - Alternativa eficiente para espacios grandes de hiperparámetros
   - Balance entre tiempo de cómputo y calidad de resultados
   - Recomendado para optimización de múltiples hiperparámetros

4. **Validación Cruzada**:
   - Fundamental para evitar overfitting
   - Proporciona estimaciones más confiables del rendimiento
   - La estratificación es crucial para datasets desbalanceados

## 📚 Referencias

- Dataset: [UCI Machine Learning Repository - Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/abalone)
- Documentación: [Scikit-learn - Model Selection](https://scikit-learn.org/stable/model_selection.html)

## 👤 Autor

**Leo Alvarez**
- Proyecto académico para Digital House Data Science

## 📄 Licencia

Este proyecto es parte de un ejercicio académico.

---

⭐ **Nota**: Este notebook está completamente documentado con explicaciones detalladas en celdas markdown antes de cada bloque de código.
