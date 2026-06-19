# Apuntes — Módulo 01: Introducción a Machine Learning

> Apuntes sintetizados y mejorados de la teoría del módulo. Para el repaso global del programa ver [`../APUNTES-DATA-SCIENCE.md`](../APUNTES-DATA-SCIENCE.md).

## 1. ¿Qué es Machine Learning?

ML es la rama de la IA que crea algoritmos capaces de **aprender una tarea a partir de datos**, sin programar reglas explícitas.

| Concepto | Alcance |
|----------|---------|
| Inteligencia Artificial | Imitar comportamiento inteligente humano (campo amplio). |
| Machine Learning | Aprender de los datos/experiencia (subconjunto de IA). |
| Deep Learning | ML basado en redes neuronales profundas. |

## 2. Datasets, features y labels

- **Dataset:** filas = observaciones, columnas = variables.
- **Features (`X`):** variables de entrada. **Label/target (`y`):** lo que se predice.
- Cantidad y calidad de datos determinan el techo del modelo.

## 3. Tipos de aprendizaje

```
Supervisado (con etiquetas)      No supervisado (sin etiquetas)
├── Regresión   → y continuo     ├── Clustering              → agrupar
└── Clasificación → y categórico └── Reducción de dimensión  → comprimir
```
Otros: semi-supervisado y aprendizaje por refuerzo.

## 4. Ciclo de vida de un proyecto de ML

1. Definición del problema (negocio, objetivos, métricas, variables).
2. Preparación de datos (EDA, limpieza, features).
3. Selección y entrenamiento del modelo (+ hiperparámetros).
4. Evaluación (métricas según el problema).
5. Optimización y refinamiento (iterativo).
6. Implementación en producción.
7. Monitoreo y mantenimiento (drift, reentrenamiento).

## 5. Preparación de datos y EDA

EDA = estadística descriptiva + visualizaciones para entender distribución, relaciones, outliers y faltantes. Tareas: imputación, encoding de categóricas, escalado, tratamiento de outliers, balanceo.

## 6. Clasificadores lineales

Decisión por combinación lineal: `f(x) = wᵀx + b`. La frontera es un **hiperplano**; se clasifica por el signo de `f(x)`.

| Modelo | Idea |
|--------|------|
| Perceptrón | Clasificador lineal más simple; ajusta pesos por error. |
| Regresión logística | **Clasifica** modelando `P(y=1\|X)` con la sigmoide `σ(z)=1/(1+e^−z)`, acotando a [0,1]. |
| SVM | Hiperplano que **maximiza el margen** entre clases. |

No se usa regresión lineal para clasificar porque daría valores fuera de [0,1].

## 7. Árboles de decisión (CART)

Estructura raíz → nodos internos (preguntas) → hojas (predicción). Ramifica eligiendo la variable que deja subnodos más **puros**.

- **Gini:** `1 − Σ pᵢ²` · **Entropía:** `−Σ pᵢ·log₂(pᵢ)`.

> **Corrección al material:** un nodo **puro tiene Gini = 0** (no "Gini alto = homogéneo", como decían las slides). El árbol **minimiza** la impureza.

Hiperparámetros: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`.
Ventajas: interpretable, tolera outliers/faltantes, no paramétrico. Desventajas: **sobreajuste**, inestabilidad → se mitigan con ensambles.

## 8. Métricas de clasificación

Matriz de confusión (TP/FN/FP/TN):

| Métrica | Fórmula | Cuándo |
|---------|---------|--------|
| Accuracy | (TP+TN)/Total | Global; engaña con desbalance. |
| Precisión | TP/(TP+FP) | Costo alto de FP. |
| Recall | TP/(TP+FN) | Costo alto de FN. |
| F1 | 2·P·R/(P+R) | Equilibrio P/R. |

ROC = TPR vs FPR; **AUC** = área (1 perfecto, 0.5 azar). Con desbalance: priorizar F1/AUC/recall.

## 9. Aprendizaje no supervisado

- **Clustering** (K-Means): agrupar por similitud. Usos: segmentación, anomalías.
- **Reducción de dimensión** (PCA): comprimir conservando información; visualizar, reducir ruido.

## 10. Evaluación de modelos

- **Overfitting:** memoriza train (alta varianza). **Underfitting:** demasiado simple (alto sesgo).
- Split train/test y **validación cruzada (k-fold)**. Buscar el mínimo del error de generalización (trade-off sesgo-varianza).

## 11. XGBoost y ensambles

- **Bagging** (Random Forest): árboles en paralelo, promedia → baja varianza.
- **Boosting** (XGBoost): árboles secuenciales que corrigen errores previos → baja sesgo.
- **XGBoost:** gradient boosting optimizado, con regularización; estándar en datos tabulares.
