# 🍷 Clasificación de Vinos con Árboles de Decisión

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de **Machine Learning** para clasificación de vinos utilizando **Árboles de Decisión** (Decision Trees) con el dataset de vinos de scikit-learn. 

### 🎯 Objetivo Principal
Clasificar vinos en **3 categorías distintas** basándose en **13 características químicas** mediante un algoritmo de aprendizaje supervisado.

### 🔬 Metodología
- Implementación de un clasificador Decision Tree
- División de datos en conjuntos de entrenamiento (75%) y prueba (25%)
- Visualización del árbol de decisión para interpretación de resultados
- Evaluación del modelo mediante métricas de precisión

## 📊 Dataset

- **Nombre**: Wine Recognition Dataset (scikit-learn)
- **Características**: 13 atributos químicos
  - Alcohol
  - Ácido málico
  - Cenizas
  - Alcalinidad de las cenizas
  - Magnesio
  - Fenoles totales
  - Flavonoides
  - Fenoles no flavonoides
  - Proantocianinas
  - Intensidad de color
  - Matiz
  - OD280/OD315 de vinos diluidos
  - Prolina
- **Clases**: 3 tipos de vinos (cultivares diferentes)
- **Total de muestras**: 178 observaciones
- **Balanceo**: Dataset relativamente balanceado entre las 3 clases

## 🛠️ Tecnologías Utilizadas

| Librería | Versión Recomendada | Propósito |
|----------|---------------------|-----------|
| **Python** | 3.8+ | Lenguaje de programación base |
| **NumPy** | 1.21+ | Operaciones numéricas y manejo de arrays |
| **Pandas** | 1.3+ | Manipulación y análisis de datos |
| **Scikit-learn** | 1.0+ | Algoritmos de Machine Learning y preprocesamiento |
| **Matplotlib** | 3.4+ | Visualización de datos y gráficos |
| *📁 Estructura del Proyecto

```
Ejercicio 1/
│
├── ejercicio1.ipynb           # 📓 Notebook principal con el análisis completo
│                               #    Incluye celdas markdown explicativas
│
├── README.md                  # 📖 Este archivo - Documentación general
│
├── explicacion_librerias.md  # 📚 Explicación detallada de las librerías utilizadas
│💻 Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior instalado
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

### Instalación de Dependencias

#### Opción 1: Instalación individual
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

#### Opción 2: Instalación desde requirements.txt (recomendado)
```bash
# Crear archivo requirements.txt con:
# n📈 Resultados Esperados

El modelo de Árbol de Decisión con profundidad máxima de 2 niveles proporciona:

- ✅ **Alta interpretabilidad**: El árbol es fácil de entender y visualizar
- 📊 **Buena precisión**: Accuracy que supera el 85% (verificar en notebook)
- 🎯 **Clasificación multiclase**: Distingue entre 3 tipos de vinos
- 🚀 **Entrenamiento rápido**: Modelo ligero y eficiente

### Métricas de Evaluación
La precisión (accuracy) del modelo se calcula comparando las predicciones con las etiquetas reales del conjunto de prueba. El resultado específico se encuentra en la última celda del notebook.

## ⚙️ Características Técnicas del Modelo

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Algoritmo** | DecisionTreeClassifier | Clasificador de scikit-learn |
| **Profundidad máxima** | 2 niveles | Evita sobreajuste, mejora generalización |
| **División de datos** | 75% / 25% | Train-test split |
| **Random state** | 1 | Garantiza reproducibilidad |
| **Criterio** | Gini (default) | Medida de impureza para divisiones |
🎓 Aprendizajes Clave

Este proyecto permite comprender:

1. **Algoritmos de clasificación supervisada**: Cómo funcionan los árboles de decisión
2. **Preprocesamiento de datos**: División train-test y exploración de datos
3. **Evaluación de modelos**: Métricas de rendimiento y validación
4. **Visualización**: Representación gráfica de modelos de ML
5. **Scikit-learn**: Uso de la biblioteca estándar de ML en Python

## 🔄 Posibles Mejoras

Ideas para extender este proyecto:

- [ ] Probar diferentes valores de `max_depth` y comparar resultados
- [ ] Implementar validación cruzada (cross-validation)
- [ ] Agregar más métricas: precision, recall, F1-score, matriz de confusión
- [ ] Comparar con otros algoritmos: Random Forest, SVM, KNN
- [ ] Realizar feature importance analysis
- [ ] Implementar grid search para optimización de hiperparámetros
- [ ] Agregar visualizaciones adicionales (distribuciones, correlaciones)

## 📚 Referencias

- [Documentación de Scikit-learn - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Wine Dataset UCI](https://archive.ics.uci.edu/ml/datasets/wine)
- [Árboles de Decisión - Teoría](https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n)

## 🤝 Contribuciones

Este es un proyecto educativo. Si encuentras errores o tienes sugerencias:
1. Abre un issue
2. Propón mejoras
3. Comparte tus resultados

## 📄 Licencia

Proyecto educativo - Digital House Data Science

## 👤 Autor

**Digital House Data Science - Ejercicio 1**

📅 **Fecha**: Diciembre 2025

---

⭐ Si este proyecto te fue útil, no olvides darle una estrella!

💡 **Tip**: Experimenta modificando los parámetros del modelo para ver cómo afectan los resultados.
## 📊 Visualización

El proyecto incluye una visualización completa del árbol de decisión que muestra:

- 🌳 **Estructura del árbol**: Nodos de decisión y hojas
- 📏 **Características clave**: Variables más importantes para la clasificación
- 🔀 **Reglas de decisión**: Umbrales de división en cada nodo
- 🎨 **Codificación por colores**: 
  - Colores diferentes representan las 3 clases de vinos
  - Intensidad del color indica la pureza de la clasificación
- 📊 **Distribución de muestras**: Cantidad de ejemplos en cada nodo
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## 🚀 Ejecución

### Método 1: Jupyter Notebook Clásico
```bash
# Navegar al directorio del proyecto
cd "Digital House Data Science/Ejercicio 1"

# Iniciar Jupyter Notebook
jupyter notebook

# Se abrirá automáticamente en tu navegador
# Haz clic en ejercicio1.ipynb
```

### Método 2: JupyterLab (interfaz moderna)
```bash
jupyter lab
```

### Método 3: Visual Studio Code
1. Instalar la extensión "Jupyter" en VS Code
2. Abrir el archivo `ejercicio1.ipynb`
3. Seleccionar el kernel de Python adecuado
4. Ejecutar las celdas secuencialmente con `Shift + Enter`

### ⚠️ Importante
- Ejecutar las celdas en **orden secuencial** de arriba hacia abajo
- Cada celda depende de las anteriores
- Las celdas markdown proporcionan contexto y explicaciones
9. **Evaluación**: Cálculo de métricas de rendimiento README.md                  # Este archivo
├── explicacion_librerias.md  # Explicación detallada de las librerías
└── explicacion_codigo.md     # Análisis paso a paso del código
```

## Instalación

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Ejecución

1. Abre Jupyter Notebook:
```bash
jupyter notebook
```

2. Navega hasta `ejercicio1.ipynb` y ejecuta las celdas secuencialmente.

## Resultados

El modelo de Árbol de Decisión con profundidad máxima de 2 niveles logra clasificar los vinos con una precisión que se puede consultar en la última celda del notebook.

## Características del Modelo

- **Algoritmo**: Decision Tree Classifier
- **Profundidad máxima**: 2 niveles
- **División de datos**: 75% entrenamiento, 25% prueba
- **Random state**: 1 (para reproducibilidad)

## Visualización

El proyecto incluye visualización del árbol de decisión, mostrando:
- Características utilizadas en cada nodo
- Decisiones de clasificación
- Colores que representan las clases

## Autor

Leo

## Fecha

Diciembre 2025
