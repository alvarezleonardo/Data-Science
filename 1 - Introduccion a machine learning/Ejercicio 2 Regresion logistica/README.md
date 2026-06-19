# Ejercicio 2: Clasificadores Lineales - Regresión Logística

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de **Regresión Logística** para predecir el riesgo de impago (default) de clientes basándose en sus características financieras. El análisis utiliza el dataset Default que contiene información sobre balance de tarjetas, ingresos y condición de estudiante.

La regresión logística es un algoritmo de **clasificación supervisada** que modela la probabilidad de que una observación pertenezca a una clase determinada. A pesar de su nombre, es un modelo de clasificación, no de regresión, y utiliza la función sigmoide para transformar una combinación lineal de variables en probabilidades entre 0 y 1.

## 🎯 Objetivo

Desarrollar un modelo predictivo que permita identificar clientes con alto riesgo de default utilizando clasificadores lineales, específicamente Regresión Logística. Este modelo puede ayudar a instituciones financieras a:
- Tomar decisiones informadas sobre aprobación de créditos
- Establecer límites de crédito apropiados
- Identificar clientes que requieren monitoreo adicional
- Reducir pérdidas por impagos

## 📊 Dataset

**Archivo:** `Default.csv`

El dataset contiene información de clientes y su historial de pagos. Los datos están desbalanceados, con aproximadamente **3.33%** de clientes que han hecho default y **96.67%** que han cumplido con sus pagos.

### Variables del Dataset:

#### Variable Objetivo (Target):
- **default**: `Yes` / `No` - Indica si el cliente ha incumplido con sus pagos

#### Variables Predictoras (Features):
- **student**: Categórica (`Yes` / `No`) - Indica si el cliente es estudiante
- **balance**: Numérica continua - Saldo promedio en la tarjeta de crédito (en dólares)
- **income**: Numérica continua - Ingreso anual del cliente (en dólares)

### Características del Dataset:
- **Tamaño**: ~10,000 observaciones
- **Formato**: CSV con separador de tabulación (`\t`)
- **Desbalanceo de clases**: Alta proporción de clase negativa (No default)

## 🛠️ Tecnologías y Librerías

El proyecto utiliza las siguientes librerías de Python:

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: 
  - `LogisticRegression`: Modelo de clasificación
  - `train_test_split`: División de datos
  - `accuracy_score`: Métrica de evaluación
  - `confusion_matrix`: Matriz de confusión

## 📁 Estructura del Proyecto

```
Ejercicio 2 Clasificadores lineales/
│
├── ejercicio RegLogistica.ipynb    # Notebook principal con análisis completo
├── Default.csv                     # Dataset de entrada (get_dummies)
- **Eliminación de multicolinealidad**: Uso de `drop_first=True` para evitar la trampa de variables dummy (dummy variable trap)
  - Si tenemos 2 categorías, solo necesitamos 1 variable dummy
  - La segunda categoría está implícita: si student_Yes = 0, entonces student_No = 1
- **Integración de variables**: Concatenación de variables dummy al dataset original
```

## 🔍 Metodología

### 1. Carga y Exploración de Datos
- Importación del dataset Default.csv
- Análisis exploratorio de la distribución de clases

### 2. Preprocesamiento
- **Codificación de variables categóricas**: Conversión de la variable `student` usando One-Hot Encoding
- **Eliminación de multicolinealidad**: Uso de  (7,500 obs.), 25% prueba (2,500 obs.)
  - `stratify=y`: Mantiene la proporción de clases en ambos conjuntos
  - `random_state=12`: Garantiza reproducibilidad de resultados
- **Validación de proporciones**: Verificación que ambos conjuntos mantienen ~3.33% de defaults
- **Modelo**: Regresión Logística sin penalización (`penalty=None`)
  - No se aplica regularización L1 (Lasso) ni L2 (Ridge)
  - El modelo aprende directamente de los datos sin restricciones en los coeficientes
- **Variables predictoras**: balance, income, student_Yes
- **Variable objetivo**: default (Yes/No)
- **Función del modelo**: 
  - Formula: (TP + TN) / (TP + TN + FP + FN)
  - Métrica simple pero puede ser engañosa con datos desbalanceados
- **Matriz de Confusión**: Análisis detallado del desempeño
  ```
                   Predicho No    Predicho Yes
  Real No          TN              FP (Error Tipo I)
  Real Yes         FN (Error Tipo II)  TP
  ```
  - **TN (Verdaderos Negativos)**: Correctamente predijo "No default"
  - **FP (Falsos Positivos)**: Predijo default incorrectamente (pierde negocio)
  - **FN (Falsos Negativos)**: No detectó default (pierde dinero) ⚠️ MÁS COSTOSO
  - Intercepto (β₀)**: Valor base de log-odds cuando todas las variables son 0
  - Representa el logaritmo del odds ratio de default en el caso base
- **Coeficientes del modelo**: Impacto de cada variable en el log-odds
  - **β₁ (balanc Esperados

El notebook presenta un análisis completo que incluye:

### Exploración de Datos
- Distribución de clases: ~3.33% default, ~96.67% no default
- Verificación de proporciones en conjuntos de entrenamiento y prueba

### Métricas de Rendimiento
- **Accuracy**: Porcentaje general de aciertos (típicamente alto debido al desbalanceo)
- **Matriz de Confusión**: Desglose detallado de TP, TN, FP, FN

### Análisis de Coeficientes
- **Balance**: Coeficiente positivo esperado (mayor saldo → mayor riesgo)
- **Income**: Coeficiente negativo esperado (mayor ingreso → menor riesgo)
- **Student_Yes**: Coeficiente que depende de las características del dataset

**Python 3.7+** con las siguientes librerías:

```bash
pip install pandas numpy scikit-learn jupyter
```

O si usas Anaconda:

```bash
conda install pandas numpy scikit-learn jupyter
```

### Datos
- El dataset debe estar en formato **CSV separado por tabulaciones** (`\t`)
- Los datos están **desbalanceados**: solo ~3.33% son defaults
- Se utiliza **división estratificada** (`stratify=y`) para mantener proporciones de clases

### Modelo
- El modelo **no incluye regularización** (`penalty=None`) para este ejercicio básico
- En producción, considerar usar `penalty='l1'` o `penalty='l2'` para evitar overfitting
- El **umbral de decisión por defecto es 0.5**, pero puede ajustarse según el caso de uso

### Interpretación
- Los **coeficientes positivos** aumentan la probabilidad de default
- Los **coeficientes negativos** disminuyen la probabilidad de default
- La **magnitud absoluta** indica la fuerza de la relación
- Para obtener el **efecto multiplicativo** sobre el odds, calcular: e^(coeficiente)

### Limitaciones
- **Accuracy no es suficiente** con datos desbalanceados: puede dar ~97% solo prediciendo "No" siempre
- Considerar métricas adicionales: **Precision, Recall, F1-Score, ROC-AUC**
- El modelo asume **relación lineal** entre variables y log-odds
- No captura **interacciones complejas** entre variables (usar modelos más avanzados si es necesario)
3. 🎓 Conceptos Clave Aprendidos

- **Clasificación binaria** con Regresión Logística
- **Codificación de variables categóricas** (One-Hot Encoding)
- **Prevención de multicolinealidad** (drop_first)
- **División estratificada** de datos desbalanceados
- **Interpretación de coeficientes** en log-odds
- **Matriz de confusión** y tipos de errores
- **Trade-off entre métricas** según contexto de negocio

## 🔧 Posibles Mejoras y Extensiones

1. **Métricas adicionales**: Implementar Precision, Recall, F1-Score, ROC-AUC
2. **Visualizaciones**: Agregar gráficos de distribuciones, ROC curves, feature importance
3. **Regularización**: Experimentar con L1 (Lasso) y L2 (Ridge) penalties
4. **Ajuste de umbral**: Optimizar el threshold según costos de FP vs FN
5. **Validación cruzada**: Implementar k-fold cross-validation
6. **Feature engineering**: Crear nuevas variables o interacciones
7. **Comparación de modelos**: Probar otros clasificadores (Random Forest, SVM, XGBoost)
8. **Balanceo de clases**: Técnicas como SMOTE o class_weight

## 📚 Referencias y Recursos

### Documentación Oficial
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Logistic Regression - Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Model Evaluation - Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Lecturas Recomendadas
- "An Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani (Capítulo 4)
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- [Understanding Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)

### Tutoriales
- [Scikit-learn Logistic Regression Tutorial](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Confusion Matrix Explained](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

**Curso:** Data Science - Digital House  
**Tema:** Clasificadores Lineales - Regresión Logística  
**Nivel:** Intermedio
1. Importación de librerías
2. Carga de datos
3. Exploración de datos
4. Preprocesamiento (variables dummy)
5. División de datos (train/test)
6. Entrenamiento del modelo
7. Predicciones
8. Evaluación (accuracy, matriz de confusión)
9. Interpretación (coeficientesión multiplicativaisis de confianza del modelo (predict_proba)
  - Útil para establecer umbrales de decisión personalizados
  - Por defecto: umbral = 0.5
- **Validación de proporciones**: Verificación de balance de clases en ambos conjuntos

### 4. Entrenamiento del Modelo
- Modelo: **Regresión Logística sin penalización** (`penalty=None`)
- Variables predictoras: balance, income, student_Yes
- Variable objetivo: default

### 5. Evaluación del Modelo
- **Accuracy (Exactitud)**: Porcentaje de predicciones correctas
- **Matriz de Confusión**: Análisis detallado de:
  - Verdaderos Positivos (TP)
  - Verdaderos Negativos (TN)
  - Falsos Positivos (FP)
  - Falsos Negativos (FN)
- **Probabilidades de predicción**: Análisis de confianza del modelo

### 6. Interpretación de Resultados
- **Coeficientes del modelo**: Interpretación del impacto de cada variable
- **Intercepto**: Valor base de log-odds

## 📈 Resultados

El notebook presenta un análisis completo con:
- Visualización de la distribución de clases
- Métricas de evaluación del modelo
- Coeficientes interpretables para cada variable predictora
- Matriz de confusión para análisis detallado de errores

## 🚀 Cómo Usar

### Requisitos Previos

```bash
pip install pandas numpy scikit-learn
```

### Ejecución

1. Asegúrate de tener el archivo `Default.csv` en el mismo directorio
2. Abre el notebook `ejercicio RegLogistica.ipynb`
3. Ejecuta las celdas secuencialmente (Shift + Enter)

## 📝 Notas Importantes

- El dataset debe estar en formato CSV separado por tabulaciones (`\t`)
- Se utiliza división estratificada para mantener las proporciones de clases
- El modelo no incluye penalización (regularización) para este ejercicio básico
- Los coeficientes positivos aumentan la probabilidad de default, los negativos la disminuyen

## 👨‍💻 Autor

**Leo**  
Ejercicio 2 - Clasificadores Lineales

## 📅 Fecha

22 de diciembre de 2025

## 📚 Referencias

- Scikit-learn Documentation: https://scikit-learn.org/
- Logistic Regression Theory: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

---

**Curso:** Data Science - Digital House  
**Tema:** Clasificadores Lineales - Regresión Logística
