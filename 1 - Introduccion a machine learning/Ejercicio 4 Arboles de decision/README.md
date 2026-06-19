# Clasificación de Películas con Árboles de Decisión 🎬🏆

## Autor
**Leo Alvarez**

## Descripción del Proyecto

Este proyecto implementa un análisis completo de clasificación basado en **árboles de decisión** para predecir si una película ganará un Oscar para el equipo técnico (`Start_Tech_Oscar`). El análisis incluye:

- Comparación de criterios de división (Gini vs Entropía)
- Análisis de importancia de características
- Selección de características más relevantes
- Optimización de modelos mediante simplificación
- Visualización detallada de los árboles de decisión

El proyecto utiliza características de películas como tiempo de duración, disponibilidad en 3D y género cinematográfico para realizar predicciones precisas.

## Estructura del Proyecto

```
├── ejercicio 4.ipynb                    # Notebook principal con el análisis
├── Movie_classification.csv             # Dataset original
├── Data/
│   ├── Movie_Classification_train_x.csv # Conjunto de entrenamiento (features)
│   ├── Movie_Classification_train_y.csv # Conjunto de entrenamiento (target)
│   ├── Movie_Classification_test_x.csv  # Conjunto de prueba (features)
│   └── Movie_Classification_test_y.csv  # Conjunto de prueba (target)
└── README.md                            # Este archivo
```

## Tecnologías Utilizadas

- **Python 3.x**
- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualización de gráficos
- **seaborn**: Visualización estadística
- **scikit-learn**: Implementación del modelo de Machine Learning

## Metodología Detallada

### 1. Carga y Exploración de Datos
- Carga del dataset `Movie_classification.csv`
- Análisis exploratorio inicial: dimensiones, tipos de datos, primeras observaciones
- Identificación de valores faltantes y análisis de distribuciones

### 2. Preprocesamiento de Datos
- **Imputación de valores faltantes**: Valores nulos en `Time_taken` reemplazados con la media
- **Codificación de variables categóricas**: 
  - One-Hot Encoding aplicado a `3D_available` y `Genre`
  - Uso de `drop_first=True` para evitar multicolinealidad
- **Separación de features y target**: 
  - X: Todas las características (features)
  - y: Variable objetivo `Start_Tech_Oscar` (binaria: 0/1)

### 3. División del Dataset
- **Train/Test Split**: 70% entrenamiento / 30% prueba
- **Random State**: 42 (garantiza reproducibilidad)
- Exportación de conjuntos a CSV para documentación y reutilización

### 4. Modelado con Árboles de Decisión

#### 4.1 Modelo con Criterio Gini (Baseline)
- **Algoritmo**: DecisionTreeClassifier con criterio Gini
- **Configuración**: Parámetros por defecto, sin restricciones de profundidad
- Análisis de importancia de características
- Evaluación con matriz de confusión y accuracy

#### 4.2 Modelo con Criterio de Entropía
- **Algoritmo Principales

### Hallazgos Clave

1. **Comparación de Criterios**:
   - Ambos modelos (Gini y Entropía) mostraron rendimiento similar
   - La estructura de los árboles difiere ligeramente entre criterios
   - Algunas características fueron seleccionadas de forma diferente

2. **Importancia de Características**:
   - Se identificaron las características más relevantes para predecir Oscars técnicos
   - Varias características mostraron importancia cero (no utilizadas por el modelo)
   - Las 4 características principales capturan la mayor parte del poder predictivo

3. **Modelos Simplificados**:
   - Los modelos con solo 4 características mantienen alto acuerdo con los modelos completos
   - La limitación de profundidad (`max_depth=7`) mejora la interpretabilidad
   - Menor riesgo de overfitting sin sacrificar rendimiento significativo

4. **Métricas de Evaluación**:
   - Matrices de confusión detalladas para cada modelo
   - Accuracy scores calculados para todas las variantes
   - Alto nivel de reproducibilidad en los resultados

### Visualizaciones Generadas

- ✅ Árbol de decisión completo (criterio Gini)
- ✅ Árbol de decisión completo (criterio Entropía)
- ✅ Árbol simplificado con 4 características (Gini)
- ✅ Árbol simplificado con 4 características (Entropía)
- ✅ Tablas de importancia de características
- ✅ Matrices de confusión para todos los modeloscada modelo
- Comparación de las características seleccionadas por ambos criterios

#### 4.4 Modelos Simplificados
- Entrenamiento de nuevos modelos usando solo las 4 mejores características
- **Limitación de profundidad**: `max_depth=7` para reducir overfitting
- Comparación de predicciones entre modelos completos y simplificados

### 5. Evaluación y Comparación
- **Métricas utilizadas**:
  - Matriz de confusión (verdaderos positivos, falsos positivos, etc.)
  - Accuracy score (exactitud de clasificación)
- **Comparaciones realizadas**:
  - Gini vs Entropía (modelos completos)
  - Modelos completos vs simplificados
  - Análisis de acuerdo entre predicciones
- **Visualizaciones**:
  - Árboles de decisión completos (ambos criterios)
  - Árboles simplificados con 4 características

## Resultados

El modelo de árbol de decisión fue entrenado exitosamente y evaluado con las siguientes métricas:
- Se visualizó la matriz de confusión para analizar los errores de clasificación
- Se calculó el accuracy score para medir el rendimiento general
- Se identificaron las características más importantes para la predicción
- Se generó una visualización completa del árbol de decisión
Conceptos Clave del Proyecto

### Árboles de Decisión
Algoritmo de Machine Learning que divide los datos mediante preguntas sucesivas, creando una estructura de árbol donde cada nodo representa una decisión y las hojas contienen las predicciones finales.

### Criterio Gini
Mide la impureza de un nodo calculando la probabilidad de clasificación incorrecta. Busca divisiones que minimicen esta impureza.

### Criterio de Entropía (Information Gain)
Mide la cantidad de información necesaria para describir la distribución de clases. Busca divisiones que maximicen la ganancia de información.

### Importancia de Características (Feature Importance)
Métrica que indica cuánto contribuye cada característica a las predicciones del modelo, basándose en la reducción de impureza que proporciona.

### One-Hot Encoding
Técnica de codificación que convierte variables categóricas en columnas binarias (0/1), permitiendo que el modelo procese datos no numéricos.

## Próximos Pasos y Mejoras Potenciales

### Optimización del Modelo
- ✨ **GridSearchCV**: Búsqueda exhaustiva de los mejores hiperparámetros
- ✨ **Cross-Validation**: Validación cruzada para mejor estimación del rendimiento
- ✨ **Poda del árbol**: Técnicas de pre-poda y post-poda para evitar overfitting

### Algoritmos Alternativos
- 🌳 **Random Forest**: Conjunto de árboles de decisión para mayor robustez
- 🚀 **Gradient Boosting**: XGBoost, LightGBM para mejor rendimiento
- 🎯 **Ensemble Methods**: Combinación de múltiples modelos

### Análisis Avanzado
- 📊 **Métricas adicionales**: Precision, Recall, F1-Score, ROC-AUC
- 🔍 **Análisis de sesgo**: Evaluación de equidad en las predicciones
- 🎨 **Feature Engineering**: Creación de nuevas características derivadas
- 📈 **Análisis de curvas de aprendizaje**: Diagnóstico de overfitting/underfitting
pip install numpy pandas seaborn matplotlib scikit-learn
```

3. **Abrir el notebook**:
```bash
jupyter notebook "ejercicio 4.ipynb"
```

4. **Ejecutar las celdas secuencialmente** para reproducir el análisis completo

## Archivos Generados

El notebook genera automáticamente los siguientes archivos en el directorio `Data/`:
- `Movie_Classification_train_x.csv`: Features de entrenamiento
- `Movie_Classification_train_y.csv`: Target de entrenamiento
- `Movie_Classification_test_x.csv`: Features de prueba
- `Movie_Classification_test_y.csv`: Target de prueba

## Próximos Pasos

Posibles mejoras al proyecto:
- Optimización de hiperparámetros con GridSearchCV
- Prueba de otros algoritmos (Random Forest, Gradient Boosting)
- Análisis de overfitting y técnicas de poda del árbol
- Feature engineering adicional
- Validación cruzada para una evaluación más robusta

## Contacto

**Leo Alvarez**
