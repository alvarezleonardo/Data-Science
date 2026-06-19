# Análisis de Clustering de Clientes de Mall 🛍️

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis de clustering utilizando el algoritmo **K-Means** sobre un dataset de clientes de un centro comercial. El objetivo es segmentar a los clientes en grupos distintos basándose en sus patrones de ingreso y gasto, permitiendo desarrollar estrategias de marketing personalizadas.

## 🎯 Objetivos

- Identificar segmentos de clientes con comportamientos similares
- Determinar el número óptimo de clusters
- Visualizar y analizar los grupos identificados
- Proporcionar insights para estrategias de negocio

## 📁 Estructura del Proyecto

```
Ejercicio 5 Clustering/
│
├── clustering.ipynb          # Notebook principal con el análisis
├── mall_customers.csv         # Dataset de clientes
└── README.md                  # Documentación del proyecto
```

## 📊 Dataset

El archivo `mall_customers.csv` contiene información de 200 clientes con las siguientes características:

| Variable | Descripción |
|----------|-------------|
| CustomerID | Identificador único del cliente |
| Gender | Género del cliente (Male/Female) |
| Age | Edad del cliente |
| Annual Income (k$) | Ingreso anual en miles de dólares |
| Spending Score (1-100) | Puntuación de gasto asignada por el mall |

## 🔧 Tecnologías y Librerías

- **Python 3.x**
- **pandas**: manipulación y análisis de datos
- **numpy**: operaciones numéricas
- **scikit-learn**: algoritmos de machine learning (K-Means, StandardScaler)
- **seaborn**: visualización estadística
- **matplotlib**: generación de gráficos

## 📝 Instalación

Para ejecutar este proyecto, necesitas instalar las siguientes librerías:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

O usando el archivo requirements.txt (si está disponible):

```bash
pip install -r requirements.txt
```

## 🚀 Uso

1. Clona o descarga este repositorio
2. Asegúrate de tener instaladas todas las dependencias
3. Abre el notebook `clustering.ipynb` en Jupyter Notebook o VS Code
4. Ejecuta las celdas secuencialmente

## 📈 Metodología

### 1. Exploración de Datos
- Carga y visualización inicial del dataset
- Análisis de distribuciones y correlaciones mediante pairplot

### 2. Preprocesamiento
- Selección de features: `Income` y `Spending Score`
- Normalización usando StandardScaler para igualar escalas

### 3. Clustering con K-Means
- Aplicación del algoritmo K-Means
- Identificación de centroides y asignación de clusters

### 4. Determinación del Número Óptimo de Clusters

#### Método del Codo (Elbow Method)
- Evalúa la inercia (suma de distancias cuadráticas) para k=2 a k=9
- Identifica el punto donde la mejora se reduce significativamente

#### Coeficiente de Silueta (Silhouette Score)
- Mide la calidad de los clusters
- Valores cercanos a 1 indican clusters bien definidos
- Complementa el método del codo para validación

### 5. Visualización
- Gráficos de dispersión mostrando los clusters identificados
- Representación de centroides
- Curvas de evaluación de métricas

## 📊 Resultados Esperados

El análisis típicamente identifica 5 segmentos principales de clientes:

1. **Segmento 1**: Bajos ingresos, bajo gasto
   - Clientes cautelosos con recursos limitados
   - Estrategia: Ofertas económicas, promociones especiales

2. **Segmento 2**: Bajos ingresos, alto gasto
   - Clientes jóvenes o impulsivos
   - Estrategia: Programas de fidelización, descuentos por volumen

3. **Segmento 3**: Ingresos medios, gasto medio
   - Clientes promedio estables
   - Estrategia: Productos de rango medio, servicios estándar

4. **Segmento 4**: Altos ingresos, bajo gasto
   - Clientes conservadores/ahorradores
   - Estrategia: Productos de alta calidad-precio, servicios premium selectivos

5. **Segmento 5**: Altos ingresos, alto gasto
   - Clientes premium - objetivo principal
   - Estrategia: Productos exclusivos, servicios VIP, experiencias personalizadas

## 💡 Aplicaciones Prácticas

- **Marketing Personalizado**: Diseñar campañas específicas para cada segmento
- **Optimización de Precios**: Ajustar estrategias de precios según el perfil del cliente
- **Gestión de Inventario**: Alinear el stock con las preferencias de cada grupo
- **Experiencia del Cliente**: Mejorar servicios según las necesidades de cada segmento
- **Programas de Fidelización**: Crear recompensas adaptadas a cada cluster

## 📚 Conceptos Clave

### K-Means Clustering
Algoritmo de aprendizaje no supervisado que agrupa datos en k clusters minimizando la varianza intracluster.

### StandardScaler
Técnica de normalización que transforma los datos a media 0 y desviación estándar 1, esencial para K-Means.

### Inercia (Inertia)
Suma de las distancias cuadráticas de cada punto a su centroide más cercano. Menor inercia indica clusters más compactos.

### Coeficiente de Silueta
Métrica que evalúa qué tan bien están separados los clusters (rango: -1 a 1).

## 🔍 Próximos Pasos

- Incorporar más variables (edad, género) en el análisis
- Probar otros algoritmos de clustering (DBSCAN, Hierarchical Clustering)
- Implementar visualizaciones interactivas
- Desarrollar un sistema de recomendaciones basado en los clusters
- Realizar análisis temporal de cambios en segmentos

## 👨‍💻 Autor

Leo Alvarez

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

**Nota**: Los resultados específicos pueden variar según los datos y los parámetros utilizados. Se recomienda ejecutar el análisis completo para obtener insights detallados.
