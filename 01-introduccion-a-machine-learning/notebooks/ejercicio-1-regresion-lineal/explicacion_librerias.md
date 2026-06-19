# Explicación Detallada de las Librerías Utilizadas

## 1. NumPy (Numerical Python)

```python
import numpy as np
```

**Propósito**: Librería fundamental para computación científica en Python.

**Funcionalidades clave**:
- Operaciones con arrays multidimensionales
- Funciones matemáticas de alto rendimiento
- Álgebra lineal, transformadas de Fourier
- Generación de números aleatorios

**En este proyecto**: Aunque no se usa explícitamente, es una dependencia de pandas y scikit-learn.

---

## 2. Pandas

```python
import pandas as pd
```

**Propósito**: Manipulación y análisis de datos estructurados.

**Funcionalidades clave**:
- DataFrames (tablas de datos bidimensionales)
- Series (arrays unidimensionales etiquetados)
- Lectura/escritura de múltiples formatos (CSV, Excel, SQL, etc.)
- Operaciones de filtrado, agregación y transformación

**En este proyecto**:
- Almacena las características (X) y etiquetas (Y) como DataFrames
- Permite operaciones como `.head()`, `.sample()`, `.groupby()`
- Facilita la inspección de datos

---

## 3. Seaborn

```python
import seaborn as sns
```

**Propósito**: Visualización estadística de datos basada en matplotlib.

**Funcionalidades clave**:
- Gráficos estadísticos atractivos
- Integración con pandas DataFrames
- Temas y paletas de colores predefinidos
- Visualizaciones complejas con código simple

**En este proyecto**:
- **Heatmap de matriz de confusión**: `sns.heatmap(cm, annot=True, ...)`
  - `annot=True`: Muestra valores en cada celda
  - `fmt='d'`: Formato decimal (enteros)
  - `cmap='Blues'`: Paleta de colores azules
  - `xticklabels/yticklabels`: Etiquetas de ejes
  - `cbar_kws`: Configuración de barra de colores

**Ventajas de seaborn**:
- Código más simple que matplotlib puro
- Estilos predeterminados profesionales
- Manejo automático de colores y leyendas
- Integración perfecta con pandas

**Otros gráficos útiles de seaborn** (para futuras extensiones):
- `sns.countplot()`: Distribución de clases
- `sns.pairplot()`: Relaciones entre características
- `sns.boxplot()`: Detectar outliers
- `sns.violinplot()`: Distribución de datos

---

## 4. Matplotlib

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

**Propósito**: Librería de visualización 2D más utilizada en Python.

**Funcionalidades clave**:
- Creación de gráficos (líneas, barras, dispersión, etc.)
- Personalización completa de visualizaciones
- Exportación a múltiples formatos
- Integración con Jupyter notebooks

**En este proyecto**:
- Visualiza el árbol de decisión con `plt.show()`
- `%matplotlib inline`: Comando mágico de Jupyter para mostrar gráficos en el notebook

---

## 5. Scikit-learn (sklearn)

### 5.1 Módulo `datasets`

```python
from sklearn import datasets
```

**Propósito**: Conjuntos de datos de ejemplo para Machine Learning.

**En este proyecto**:
- `datasets.load_wine()`: Carga el dataset de vinos
- Parámetros:
  - `return_X_y=True`: Retorna características y etiquetas por separado
  - `as_frame=True`: Retorna como pandas DataFrames

### 5.2 Módulo `tree`

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

**`DecisionTreeClassifier`**:
- Algoritmo de clasificación basado en árboles de decisión
- Parámetros importantes:
  - `max_depth=2`: Limita la profundidad del árbol a 2 niveles
  - Previene overfitting (sobreajuste)

**Métodos principales**:
- `.fit(X_train, Y_train)`: Entrena el modelo
- `.predict(X_test)`: Realiza predicciones
- `.tree_.node_count`: Retorna el número de nodos del árbol

**`plot_tree`**:
- Visualiza el árbol de decisión
- Parámetros:
  - `feature_names`: Nombres de las características
  - `filled=True`: Colorea los nodos según la clase
  - `class_names=True`: Muestra nombres de clases
  - `label='none'`: No muestra etiquetas en nodos
  - `impurity=False`: No muestra la impureza

### 5.3 Módulo `model_selection`

```python
from sklearn.model_selection import train_test_split
```

**Propósito**: Divide el dataset en conjuntos de entrenamiento y prueba.

**Parámetros**:
- `X, Y`: Datos a dividir
- `random_state=1`: Semilla para reproducibilidad
- Por defecto: 75% entrenamiento, 25% prueba

**Retorna**: `X_train, X_test, Y_train, Y_test`

### 5.4 Módulo `metrics`

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Propósito**: Evalúa el rendimiento del modelo con múltiples métricas.

**`accuracy_score(y_true, y_pred)`**:
- Calcula la precisión (accuracy) del modelo
- Fórmula: (Predicciones correctas) / (Total de predicciones)
- Retorna un valor entre 0 y 1 (0% a 100%)
- Ejemplo: `accuracy_score(Y_test, ypred)` → 0.95

**`classification_report(y_true, y_pred, target_names)`**:
- Genera un reporte completo con múltiples métricas
- Incluye por cada clase:
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **F1-Score**: Media armónica de precision y recall
  - **Support**: Número de muestras de cada clase
- También incluye promedios:
  - **Macro avg**: Promedio simple de todas las clases
  - **Weighted avg**: Promedio ponderado por support
- Formato de tabla legible

**`confusion_matrix(y_true, y_pred)`**:
- Crea una matriz que muestra predicciones vs valores reales
- Retorna un array numpy 2D
- **Filas**: Clases reales
- **Columnas**: Clases predichas
- **Diagonal**: Predicciones correctas
- **Fuera de diagonal**: Errores de clasificación
- Ejemplo para 3 clases:
  ```
```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO COMPLETO ML                        │
└─────────────────────────────────────────────────────────────┘

1. 📥 CARGA DE DATOS
   └─ sklearn.datasets.load_wine() → X, Y

2. 🔍 EXPLORACIÓN Y ANÁLISIS
   ├─ pandas: X.head(), X.shape, Y.groupby()
   └─ Comprender estructura y distribución

3. ✂️ DIVISIÓN DE DATOS
   └─ sklearn.model_selection.train_test_split()
      → X_train, X_test, Y_train, Y_test

4. 🎓 ENTRENAMIENTO
   ├─ sklearn.tree.DecisionTreeClassifier(max_depth=2)
   └─ modelo.fit(X_train, Y_train)

5. 🎯 PREDICCIÓN
   └─ modelo.predict(X_test) → ypred

6. 📊 VISUALIZACIÓN DEL MODELO
   ├─ sklearn.tree.plot_tree()
   └─ matplotlib.pyplot.show()

7. 📈 EVALUACIÓN CUANTITATIVA
   ├─ sklearn.metrics.accuracy_score()
   ├─ sklearn.metrics.classification_report()
   └─ sklearn.metrics.confusion_matrix()

8. 🎨 VISUALIZACIÓN DE RESULTADOS
   ├─ seaborn.heatmap() → Matriz de confusión
   └─ pandas.DataFrame() → Análisis de predicciones

9. 💡 CONCLUSIONES
   └─ Interpretar métricas y proponer mejoras
```

---

## Interacción entre Librerías

```python
# 1. NumPy (Base de todo)
#    ↓ Provee estructuras de datos eficientes
#
# 2. Pandas (Construido sobre NumPy)
#    ↓ DataFrames y Series
#
# 3. Scikit-learn (Usa NumPy/Pandas)
#    ↓ Algoritmos ML
#
# 4. Matplotlib (Visualización base)
#    ↓ Gráficos 2D
#
# 5. Seaborn (Construido sobre Matplotlib)
#    ↓ Visualizaciones estadísticas elegantes
```

**Ejemplo de integración**:
```python
# Pandas → Scikit-learn
X, Y = datasets.load_wine(return_X_y=True, as_frame=True)  # Pandas DataFrame

# Scikit-learn → NumPy
cm = confusion_matrix(Y_test, ypred)  # Retorna NumPy array

# NumPy → Seaborn → Matplotlib
sns.heatmap(cm, ...)  # Seaborn acepta NumPy array
plt.show()  # Matplotlib muestra el gráfico
```
| Métrica | ¿Qué mide? | ¿Cuándo es importante? |
|---------|-----------|------------------------|
| **Accuracy** | % total de aciertos | Clases balanceadas |
| **Precision** | De las predicciones +, cuántas correctas | Evitar falsos positivos (ej: spam) |
| **Recall** | De las reales +, cuántas detectadas | Evitar falsos negativos (ej: fraude) |
| **F1-Score** | Balance precision-recall | Clases desbalanceadas |

---

## Flujo de Trabajo con estas Librerías

1. **Carga de datos** (sklearn.datasets)
2. **Exploración** (pandas)
3. **División de datos** (sklearn.model_selection)
4. **Entrenamiento** (sklearn.tree.DecisionTreeClassifier)
5. **Predicción** (modelo entrenado)
6. **Visualización** (sklearn.tree.plot_tree, matplotlib)
7. **Evaluación** (sklearn.metrics)

---

## Versiones Recomendadas

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Instalación

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

o con un archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```
