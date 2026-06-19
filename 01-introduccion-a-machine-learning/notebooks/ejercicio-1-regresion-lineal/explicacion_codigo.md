# Explicación Paso a Paso del Código

## Celda 1: Importación de Librerías

```python
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import datasets
```

**Propósito**: Importar todas las librerías necesarias para el análisis.

**Detalles**:
- `numpy` y `pandas`: Manipulación de datos
- `seaborn` y `matplotlib`: Visualización
- `%matplotlib inline`: Muestra gráficos dentro del notebook
- `sklearn.datasets`: Acceso a datasets de ejemplo

---

## Celda 2: Carga del Dataset

```python
(X, Y) = datasets.load_wine(return_X_y=True, as_frame=True)
```

**Propósito**: Cargar el dataset de vinos.

**Detalles**:
- `X`: DataFrame con 13 características (variables independientes)
  - Ejemplo: alcohol, ácido málico, cenizas, alcalinidad, magnesio, fenoles, flavonoides, etc.
- `Y`: Serie con las etiquetas de clase (variable dependiente)
  - 3 clases: 0, 1, 2 (representan diferentes tipos de vino)
- `return_X_y=True`: Retorna X e Y por separado
- `as_frame=True`: Retorna como pandas DataFrame en lugar de arrays numpy

---

## Celda 3: Verificar Dimensiones de X

```python
print('Shape of X:', X.shape)
```

**Propósito**: Verificar el número de filas (muestras) y columnas (características).

**Resultado esperado**: `(178, 13)`
- 178 muestras de vinos
- 13 características por muestra

---

## Celda 4: Visualizar Primeras Filas

```python
X.head()
```

**Propósito**: Mostrar las primeras 5 filas del DataFrame.

**Utilidad**:
- Inspeccionar la estructura de los datos
- Verificar nombres de columnas
- Identificar el tipo de valores (flotantes, enteros, etc.)

---

## Celda 5: Dimensiones de Y

```python
Y.shape
```

**Propósito**: Verificar el tamaño del vector de etiquetas.

**Resultado esperado**: `(178,)`
- 178 etiquetas (una por cada muestra)

---

## Celda 6: Distribución de Clases

```python
Y.groupby(Y).count()
```

**Propósito**: Contar cuántas muestras hay de cada clase.

**Utilidad**:
- Verificar si el dataset está balanceado
- Entender la distribución de clases

**Interpretación**:
- Muestra cuántos vinos de cada tipo hay en el dataset

---

## Celda 7: Importar el Clasificador

```python
from sklearn.tree import DecisionTreeClassifier
```

**Propósito**: Importar el algoritmo de Árbol de Decisión.

---

## Celda 8: Crear Instancia del Modelo

```python
tree_instance = DecisionTreeClassifier(max_depth=2)
```

**Propósito**: Crear un objeto del clasificador con configuración específica.

**Parámetro importante**:
- `max_depth=2`: Limita el árbol a 2 niveles de profundidad
  - **Ventaja**: Previene overfitting
  - **Desventaja**: Puede limitar la precisión del modelo
  - Es un hiperparámetro que se puede ajustar

**¿Por qué max_depth=2?**
- Árbol simple y fácil de interpretar
- Bueno para demostración educativa
- Reduce complejidad computacional

---

## Celda 9 y 10: Exploración con Muestras

```python
X.sample(2)
Y.sample(3)
```

**Propósito**: Mostrar muestras aleatorias de los datos.

**Utilidad**:
- Verificar la calidad de los datos
- Familiarizarse con los valores típicos

---

## Celda 11: División de Datos

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
```

**Propósito**: Dividir el dataset en conjuntos de entrenamiento y prueba.

**Parámetros**:
- `random_state=1`: Semilla para reproducibilidad (siempre produce la misma división)
- División por defecto: 75% entrenamiento, 25% prueba

**Variables resultantes**:
- `X_train`: Características para entrenar (≈133 muestras)
- `X_test`: Características para probar (≈45 muestras)
- `Y_train`: Etiquetas para entrenar
- `Y_test`: Etiquetas para probar

**Importancia**:
- Evita overfitting
- Permite evaluar el modelo con datos no vistos durante el entrenamiento

---

## Celda 12: Entrenamiento del Modelo

```python
tree_instance.fit(X_train, Y_train)
```

**Propósito**: Entrenar el árbol de decisión con los datos de entrenamiento.

**Proceso interno**:
1. El algoritmo busca la mejor característica y punto de corte para dividir los datos
2. Repite el proceso recursivamente hasta alcanzar `max_depth=2`
3. Cada nodo del árbol representa una decisión basada en una característica

**Resultado**: El modelo aprende patrones de los datos de entrenamiento

---

## Celda 13: Contar Nodos

```python
tree_instance.tree_.node_count
```

**Propósito**: Obtener el número total de nodos en el árbol.

**Interpretación**:
- Con `max_depth=2`, el número máximo de nodos sería 7 (1+2+4)
- Puede ser menor si el árbol no necesita ramificarse completamente

---

## Celda 14 y 17: Visualización del Árbol

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
plot_tree(tree_instance, feature_names=X_train.columns, filled=True, 
          class_names=True, label='none', impurity=False)
plt.show()
```

**Propósito**: Visualizar gráficamente el árbol de decisión.

**Parámetros**:
- `feature_names=X_train.columns`: Muestra nombres reales de características
- `filled=True`: Colorea nodos según la clase predominante
- `class_names=True`: Muestra nombres de clases
- `label='none'`: No muestra etiquetas adicionales
- `impurity=False`: Oculta la medida de impureza (Gini)

**Interpretación del árbol**:
- **Nodos internos**: Contienen una condición (ej: "proline <= 755")
- **Nodos hoja**: Contienen la clasificación final
- **Colores**: Representan las diferentes clases de vino

---

## Celda 15: Realizar Predicciones

```python
ypred = tree_instance.predict(X_test)
ypred
```

**Propósito**: Predecir las clases para el conjunto de prueba.

**Proceso**:
1. Toma cada muestra de `X_test`
2. La pasa por el árbol de decisión entrenado
3. Retorna la clase predicha para cada muestra

**Resultado**: Array con las predicciones (valores: 0, 1, o 2)

---

## Celda 16: Inspección de Predicciones

```python
print(X_test.iloc[0:2,[0,6]])
print("Etiquetas: ", ypred[0:2])
```

**Propósito**: Comparar algunas características con sus predicciones.

**Detalles**:
- `.iloc[0:2,[0,6]]`: Selecciona las primeras 2 filas y las columnas 0 y 6
- Muestra cómo el modelo asigna etiquetas basándose en las características

**Utilidad**: Verificar manualmente que las predicciones tienen sentido

---

## Celda 18: Evaluación del Modelo - Accuracy Detallado

```python
# Calcular accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, ypred)

# Mostrar resultados detallados
print("="*50)
print("ANÁLISIS DE ACCURACY DEL MODELO")
print("="*50)
print(f"\n📊 Accuracy (Precisión): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n🎯 Predicciones correctas: {(ypred == Y_test).sum()} de {len(Y_test)}")
print(f"❌ Predicciones incorrectas: {(ypred != Y_test).sum()} de {len(Y_test)}")
print(f"\n📈 Tasa de acierto: {accuracy*100:.2f}%")
print(f"📉 Tasa de error: {(1-accuracy)*100:.2f}%")
print("="*50)
```

**Propósito**: Calcular y mostrar la precisión (accuracy) del modelo de forma detallada.

**Cálculo**:
```
Accuracy = Predicciones Correctas / Total de Predicciones
```

**Métricas mostradas**:
- **Accuracy**: Porcentaje total de aciertos
- **Predicciones correctas/incorrectas**: Conteo absoluto
- **Tasa de acierto**: Porcentaje de clasificaciones correctas
- **Tasa de error**: Porcentaje de clasificaciones incorrectas

**Interpretación**:
- Valor entre 0 y 1 (o 0% y 100%)
- Ejemplo: 0.95 = 95% de precisión
- Indica qué porcentaje de vinos fueron clasificados correctamente

**Limitaciones de accuracy**:
- Puede ser engañoso con clases desbalanceadas
- No distingue entre tipos de errores (falsos positivos vs falsos negativos)
- Por eso es importante complementar con otras métricas

---

## Celda 19: Análisis Detallado por Clase

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Reporte de clasificación completo
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print("="*60)
print(classification_report(Y_test, ypred, target_names=['Clase 0', 'Clase 1', 'Clase 2']))

# Matriz de confusión
print("\nMATRIZ DE CONFUSIÓN")
print("="*60)
cm = confusion_matrix(Y_test, ypred)
print(cm)
print("\nInterpretación:")
print("- Filas: Clases reales")
print("- Columnas: Clases predichas")
print("- Diagonal: Predicciones correctas")
```

**Propósito**: Obtener métricas detalladas por cada clase de vino.

**Classification Report incluye**:
- **Precision**: De todas las predicciones de una clase, cuántas fueron correctas
  - Fórmula: TP / (TP + FP)
- **Recall**: De todas las muestras reales de una clase, cuántas fueron detectadas
  - Fórmula: TP / (TP + FN)
- **F1-Score**: Media armónica entre precision y recall
  - Fórmula: 2 * (precision * recall) / (precision + recall)
- **Support**: Número de muestras reales de cada clase

**Matriz de Confusión**:
- Tabla que muestra predicciones correctas e incorrectas
- **Diagonal principal**: Predicciones correctas
- **Fuera de diagonal**: Errores de clasificación
- Permite identificar qué clases se confunden entre sí

**Ejemplo de interpretación**:
```
           Predicho 0  Predicho 1  Predicho 2
Real 0         15          0           0
Real 1          0          13          2
Real 2          0          1           14
```
- La clase 1 se confunde 2 veces con la clase 2
- La clase 2 se confunde 1 vez con la clase 1

---

## Celda 20: Visualización de la Matriz de Confusión

```python
# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Clase 0', 'Clase 1', 'Clase 2'],
            yticklabels=['Clase 0', 'Clase 1', 'Clase 2'],
            cbar_kws={'label': 'Número de muestras'})
plt.title('Matriz de Confusión - Árbol de Decisión', fontsize=14, fontweight='bold')
plt.ylabel('Clase Real', fontsize=12)
plt.xlabel('Clase Predicha', fontsize=12)
plt.tight_layout()
plt.show()
```

**Propósito**: Crear una visualización gráfica de la matriz de confusión.

**Parámetros del heatmap**:
- `cm`: Matriz de confusión calculada anteriormente
- `annot=True`: Muestra los números en cada celda
- `fmt='d'`: Formato decimal (enteros)
- `cmap='Blues'`: Paleta de colores azules
- `xticklabels/yticklabels`: Nombres de las clases
- `cbar_kws`: Configuración de la barra de colores

**Ventajas de la visualización**:
- Identificación rápida de patrones de error
- Colores más intensos = más muestras
- Fácil de interpretar visualmente
- Útil para presentaciones

---

## Celda 21: Comparación de Predicciones

```python
# Crear DataFrame con resultados
resultados = pd.DataFrame({
    'Clase Real': Y_test.values,
    'Clase Predicha': ypred,
    'Correcto': Y_test.values == ypred
})

print("📊 RESUMEN DE RESULTADOS:")
print("\n✅ Predicciones CORRECTAS:")
print(resultados[resultados['Correcto'] == True].head(10))

print("\n\n❌ Predicciones INCORRECTAS:")
incorrectas = resultados[resultados['Correcto'] == False]
if len(incorrectas) > 0:
    print(incorrectas)
else:
    print("¡No hay predicciones incorrectas! Accuracy del 100%")
```

**Propósito**: Crear un DataFrame para analizar predicciones individuales.

**Columnas del DataFrame**:
- `Clase Real`: Valor verdadero de Y_test
- `Clase Predicha`: Valor predicho por el modelo
- `Correcto`: Booleano indicando si la predicción fue correcta

**Utilidad**:
- Inspeccionar casos específicos de éxito o error
- Identificar patrones en las predicciones incorrectas
- Debug del modelo

**Análisis de errores**:
- Si hay pocas predicciones incorrectas: buen modelo
- Si todas las incorrectas son de la misma clase: puede haber desbalanceo
- Permite mejorar el modelo enfocándose en casos problemáticos

---

## Flujo Completo del Proyecto

```
1. PREPARACIÓN
   ├── Importar librerías
   └── Cargar dataset

2. EXPLORACIÓN DE DATOS
   ├── Verificar dimensiones (X.shape, Y.shape)
   ├── Visualizar muestras (X.head(), X.sample())
   └── Analizar distribución de clases (Y.groupby())

3. PREPARACIÓN DEL MODELO
   ├── Importar DecisionTreeClassifier
   ├── Crear instancia del clasificador (max_depth=2)
   └── Dividir datos en train/test (75%/25%)

4. ENTRENAMIENTO
   ├── Entrenar el modelo (fit())
   └── Analizar estructura (node_count)

5. VISUALIZACIÓN DEL MODELO
   ├── Graficar árbol de decisión (plot_tree)
   └── Interpretar nodos y reglas de decisión

6. PREDICCIONES
   ├── Realizar predicciones (predict())
   └── Inspeccionar predicciones individuales

7. EVALUACIÓN COMPLETA
   ├── Calcular accuracy general
   ├── Generar classification report (precision, recall, f1-score)
   ├── Crear matriz de confusión
   ├── Visualizar matriz de confusión (heatmap)
   ├── Analizar predicciones correctas e incorrectas
   └── Sacar conclusiones

8. CONCLUSIONES
   ├── Interpretar resultados
   ├── Identificar ventajas y limitaciones
   └── Proponer mejoras futuras
```

---

## Conceptos Clave de Machine Learning Aplicados

### 1. **Supervisado vs No Supervisado**
- Este es un problema de **aprendizaje supervisado**: tenemos etiquetas (Y)

### 2. **Clasificación vs Regresión**
- EMétricas de Evaluación Implementadas

### 1. **Accuracy (Precisión Global)**
- Porcentaje total de predicciones correctas
- Fácil de interpretar
- ⚠️ Puede ser engañosa con clases desbalanceadas

### 2. **Precision (Precisión por Clase)**
- De las predicciones de una clase, cuántas son correctas
- Importante cuando los falsos positivos son costosos
- Ejemplo: En diagnóstico médico

### 3. **Recall (Exhaustividad)**
- De las muestras reales de una clase, cuántas detectamos
- Importante cuando los falsos negativos son costosos
- Ejemplo: Detectar fraudes

### 4. **F1-Score**
- Balance entre precision y recall
- Útil con clases desbalanceadas
- Media armónica (penaliza valores extremos)

### 5. **Matriz de Confusión**
- Vista detallada de todos los tipos de errores
- Identifica qué clases se confunden
- Base para calcular otras métricas

---

## Posibles Mejoras y Extensiones

1. **Validación Cruzada** (Cross-Validation)
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(tree_instance, X, Y, cv=5)
   print(f"Scores: {scores}")
   print(f"Promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")
   ```

2. **Optimización de Hiperparámetros**
   ```python
   from sklearn.model_selection import GridSearchCV
   params = {
       'max_depth': [2, 3, 4, 5, 6],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
   }
   grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
   grid.fit(X_train, Y_train)
   print(f"Mejores parámetros: {grid.best_params_}")
   ```

3. **Feature Importance (Importancia de Características)**
   ```python
   importances = tree_instance.feature_importances_
   feature_importance_df = pd.DataFrame({
       'Feature': X.columns,
       'Importance': importances
   }).sort_values('Importance', ascending=False)
   print(feature_importance_df)
   ```

4. **Curva ROC y AUC** (para clasificación binaria)
   ```python
   from sklearn.metrics import roc_curve, auc
   from sklearn.preprocessing import label_binarize
   # Útil para evaluar modelos binarios
   ```

5. **Random Forest** (Mejora del Decision Tree)
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf_model = RandomForestClassifier(n_estimators=100, max_depth=3)
   rf_model.fit(X_train, Y_train)
   rf_pred = rf_model.predict(X_test)
   print(f"Accuracy RF: {accuracy_score(Y_test, rf_pred)}")
   ```

6. **Comparación de Múltiples Modelos**
   ```python
   from sklearn.svm import SVC
   from sklearn.neighbors import KNeighborsClassifier
   
   modelos = {
       'Decision Tree': DecisionTreeClassifier(max_depth=2),
       'Random Forest': RandomForestClassifier(n_estimators=100),
       'SVM': SVC(kernel='rbf'),
       'KNN': KNeighborsClassifier(n_neighbors=5)
   }
   
   for nombre, modelo in modelos.items():
       modelo.fit(X_train, Y_train)
       pred = modelo.predict(X_test)
       acc = accuracy_score(Y_test, pred)
       print(f"{nombre}: {acc:.4f}")
   ```

---

## Conclusiones del Análisis

### Ventajas del Modelo Implementado
✅ **Alta interpretabilidad**: El árbol es visual y fácil de explicar
✅ **Entrenamiento rápido**: Pocas computaciones necesarias
✅ **Buena precisión**: Con solo 2 niveles obtiene buenos resultados
✅ **No requiere normalización**: Los árboles no necesitan escalar datos
✅ **Métricas completas**: Evaluación exhaustiva del rendimiento

### Limitaciones Identificadas
⚠️ **Profundidad limitada**: max_depth=2 puede perder patrones complejos
⚠️ **Sensibilidad al split**: Diferentes random_state dan diferentes resultados
⚠️ **Overfitting potencial**: Sin validación cruzada
⚠️ **No usa todas las características**: Solo las más discriminantes

### Recomendaciones para Producción
1. Implementar validación cruzada
2. Optimizar hiperparámetros con GridSearch
3. Probar Random Forest para mayor robustez
4. Análizar feature importance
5. Guardar modelo entrenado (pickle/joblib)
6. Monitorear performance en datos nuevos

---

## Recursos Adicionales

- [Documentación Decision Trees - Scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
- [Understanding Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Wine Dataset UCI](https://archive.ics.uci.edu/ml/datasets/wine)ams = {'max_depth': [2, 3, 4, 5]}
   grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
   ```

4. **Feature Importance**
   ```python
   importances = tree_instance.feature_importances_
   ```

5. **Normalización de Datos**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
