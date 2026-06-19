# Plan de Análisis: Exploración de Datos de Negocio (EDN)

Este documento detalla el pipeline técnico completo para realizar un Análisis Exploratorio de Datos (EDA) siguiendo los lineamientos académicos, enfocándose en la calidad, la correlatividad y la reducción de dimensiones.

---

## 📋 Índice
0. [Identificación de Tipos de Variables](#0-identificación-de-tipos-de-variables)
1. [Perfilado de Calidad y Limpieza](#1-perfilado-de-calidad-y-limpieza)
2. [Análisis Univariado](#2-análisis-univariado)
3. [Análisis Bivariado y Correlatividad](#3-análisis-bivariado-y-correlatividad)
4. [Análisis Multivariado](#4-análisis-multivariado)
5. [Implementación en Python](#5-implementación-en-python)

---

## 0. Identificación de Tipos de Variables

Antes de comenzar cualquier análisis, es fundamental **identificar correctamente el tipo de cada variable** en el dataset. Esta clasificación determinará qué técnicas estadísticas y visualizaciones aplicar.

### 0.1 Taxonomía de Variables

#### 📊 Clasificación de Variables

```
Variables
│
├── CUALITATIVAS (Categóricas)
│   ├── Nominales: Sin orden inherente
│   │   └── Ejemplos: género, país, color, tipo_sangre
│   │
│   └── Ordinales: Con orden lógico
│       └── Ejemplos: nivel_educativo, satisfacción, talla (S/M/L/XL)
│
└── CUANTITATIVAS (Numéricas)
    ├── Discretas: Valores contables (enteros)
    │   └── Ejemplos: número_hijos, cantidad_productos, edad (años completos)
    │
    └── Continuas: Valores medibles (decimales)
        └── Ejemplos: peso, altura, temperatura, precio, salario
```

#### 🎯 Criterios de Identificación

| Tipo | Características | Operaciones Válidas | Python dtype |
|------|----------------|---------------------|--------------|
| **Nominal** | Categorías sin orden | Moda, frecuencias, Chi² | `object`, `category` |
| **Ordinal** | Categorías con orden | Mediana, percentiles, rangos | `category` (ordered) |
| **Discreta** | Números enteros contables | Todos los estadísticos | `int64`, `int32` |
| **Continua** | Números con decimales | Todos los estadísticos | `float64`, `float32` |

---

### 0.2 Detección Automática de Tipos de Variables

#### 🔍 Método 1: Inspección Básica con Pandas

```python
import pandas as pd
import numpy as np

def inspeccionar_dataset(df):
    """
    Inspección inicial completa del dataset
    """
    print("=" * 80)
    print(" INSPECCIÓN INICIAL DEL DATASET ".center(80, "="))
    print("=" * 80)
    
    # 1. Información general
    print(f"\n📊 DIMENSIONES")
    print(f"   Filas: {df.shape[0]:,}")
    print(f"   Columnas: {df.shape[1]}")
    print(f"   Total de celdas: {df.shape[0] * df.shape[1]:,}")
    
    # 2. Tipos de datos nativos de pandas
    print(f"\n📝 TIPOS DE DATOS (dtype)")
    print(df.dtypes.value_counts())
    
    # 3. Información detallada
    print(f"\n🔍 INFORMACIÓN DETALLADA")
    df.info()
    
    # 4. Primeras filas
    print(f"\n👀 PRIMERAS 5 FILAS")
    print(df.head())
    
    return df.dtypes

# Ejemplo de uso
df = pd.read_csv('framingham.csv')
tipos = inspeccionar_dataset(df)
```

#### 🧠 Método 2: Clasificación Inteligente de Variables

```python
def clasificar_variables(df, umbral_categorica=10, umbral_unica=0.95):
    """
    Clasifica automáticamente las variables en diferentes tipos
    
    Parámetros:
    -----------
    df : DataFrame
        Dataset a analizar
    umbral_categorica : int
        Máximo de valores únicos para considerar una variable como categórica
    umbral_unica : float
        Proporción de valores únicos para detectar identificadores
    
    Retorna:
    --------
    dict : Diccionario con la clasificación de variables
    """
    
    clasificacion = {
        'numericas_continuas': [],
        'numericas_discretas': [],
        'categoricas_nominales': [],
        'categoricas_ordinales': [],
        'binarias': [],
        'identificadores': [],
        'fechas': [],
        'texto': []
    }
    
    for columna in df.columns:
        n_unicos = df[columna].nunique()
        n_total = len(df[columna].dropna())
        prop_unica = n_unicos / n_total if n_total > 0 else 0
        dtype = df[columna].dtype
        
        # 1. IDENTIFICADORES: Casi todos los valores son únicos
        if prop_unica > umbral_unica:
            clasificacion['identificadores'].append(columna)
        
        # 2. FECHAS: dtype datetime
        elif pd.api.types.is_datetime64_any_dtype(df[columna]):
            clasificacion['fechas'].append(columna)
        
        # 3. BINARIAS: Solo 2 valores únicos
        elif n_unicos == 2:
            clasificacion['binarias'].append(columna)
        
        # 4. NUMÉRICAS
        elif pd.api.types.is_numeric_dtype(df[columna]):
            # Verificar si es entera (discreta) o continua
            if pd.api.types.is_integer_dtype(df[columna]) and n_unicos <= 50:
                clasificacion['numericas_discretas'].append(columna)
            else:
                # Verificar si tiene decimales
                tiene_decimales = (df[columna].dropna() % 1 != 0).any()
                if tiene_decimales or n_unicos > 50:
                    clasificacion['numericas_continuas'].append(columna)
                else:
                    clasificacion['numericas_discretas'].append(columna)
        
        # 5. CATEGÓRICAS
        elif pd.api.types.is_object_dtype(df[columna]) or pd.api.types.is_categorical_dtype(df[columna]):
            if n_unicos <= umbral_categorica:
                clasificacion['categoricas_nominales'].append(columna)
            else:
                clasificacion['texto'].append(columna)
    
    return clasificacion


def mostrar_clasificacion(clasificacion, df):
    """
    Muestra la clasificación de variables de forma organizada
    """
    print("\n" + "=" * 80)
    print(" CLASIFICACIÓN DE VARIABLES ".center(80, "="))
    print("=" * 80)
    
    for tipo, columnas in clasificacion.items():
        if columnas:
            print(f"\n📌 {tipo.upper().replace('_', ' ')} ({len(columnas)})")
            for col in columnas:
                n_unicos = df[col].nunique()
                n_nulos = df[col].isna().sum()
                pct_nulos = (n_nulos / len(df)) * 100
                print(f"   • {col:30} | Únicos: {n_unicos:6} | Nulos: {n_nulos:6} ({pct_nulos:5.2f}%)")


# Ejemplo de uso
clasificacion = clasificar_variables(df, umbral_categorica=10)
mostrar_clasificacion(clasificacion, df)
```

#### 📋 Método 3: Generar Dataset Resumen de Columnas

```python
def generar_resumen_columnas(df):
    """
    Genera un DataFrame completo con información de cada columna
    """
    resumen = []
    
    for columna in df.columns:
        info = {
            'Columna': columna,
            'Tipo_Python': str(df[columna].dtype),
            'Valores_Unicos': df[columna].nunique(),
            'Valores_Nulos': df[columna].isna().sum(),
            'Pct_Nulos': f"{(df[columna].isna().sum() / len(df)) * 100:.2f}%",
            'Pct_Completitud': f"{((len(df) - df[columna].isna().sum()) / len(df)) * 100:.2f}%",
            'Memoria_MB': df[columna].memory_usage(deep=True) / 1024**2
        }
        
        # Información específica según tipo
        if pd.api.types.is_numeric_dtype(df[columna]):
            info['Min'] = df[columna].min()
            info['Max'] = df[columna].max()
            info['Media'] = df[columna].mean()
            info['Mediana'] = df[columna].median()
            info['Tipo_Variable'] = 'Numérica'
            
        else:
            valores_top = df[columna].value_counts().head(3).to_dict()
            info['Valores_Frecuentes'] = str(valores_top)
            info['Tipo_Variable'] = 'Categórica'
        
        resumen.append(info)
    
    df_resumen = pd.DataFrame(resumen)
    return df_resumen


# Generar y mostrar resumen
df_resumen = generar_resumen_columnas(df)
print("\n" + "=" * 100)
print(" RESUMEN DETALLADO DE COLUMNAS ".center(100, "="))
print("=" * 100)
print(df_resumen.to_string(index=False))

# Exportar a CSV para documentación
df_resumen.to_csv('resumen_columnas.csv', index=False)
print("\n✅ Resumen exportado a 'resumen_columnas.csv'")
```

#### 🎨 Método 4: Visualización de Tipos de Variables

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_tipos_variables(clasificacion, df):
    """
    Crea visualizaciones para entender la composición del dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GRÁFICO DE PASTEL: Distribución de tipos
    tipos = {k: len(v) for k, v in clasificacion.items() if v}
    colores = plt.cm.Set3(range(len(tipos)))
    
    axes[0, 0].pie(tipos.values(), labels=tipos.keys(), autopct='%1.1f%%',
                   colors=colores, startangle=90)
    axes[0, 0].set_title('Distribución de Tipos de Variables', fontweight='bold', fontsize=14)
    
    # 2. GRÁFICO DE BARRAS: Cantidad por tipo
    axes[0, 1].barh(list(tipos.keys()), list(tipos.values()), color=colores)
    axes[0, 1].set_xlabel('Cantidad de Variables')
    axes[0, 1].set_title('Cantidad por Tipo de Variable', fontweight='bold', fontsize=14)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. HEATMAP: Valores nulos por variable
    nulos = df.isnull().sum().sort_values(ascending=False).head(20)
    if len(nulos) > 0:
        sns.barplot(x=nulos.values, y=nulos.index, ax=axes[1, 0], palette='Reds_r')
        axes[1, 0].set_xlabel('Cantidad de Valores Nulos')
        axes[1, 0].set_title('Top 20 Variables con Valores Nulos', fontweight='bold', fontsize=14)
        axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. HISTOGRAMA: Distribución de valores únicos
    valores_unicos = [df[col].nunique() for col in df.columns]
    axes[1, 1].hist(valores_unicos, bins=30, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Número de Valores Únicos')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Valores Únicos por Variable', fontweight='bold', fontsize=14)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_tipos_variables.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Visualización guardada en 'analisis_tipos_variables.png'")


# Generar visualización
visualizar_tipos_variables(clasificacion, df)
```

---

### 0.3 Detección de Variables Especiales

#### 🔍 Variables Binarias (Booleanas)

```python
def detectar_binarias(df):
    """
    Identifica variables binarias y su codificación
    """
    binarias = {}
    
    for col in df.columns:
        if df[col].nunique() == 2:
            valores = df[col].unique()
            binarias[col] = {
                'valores': valores.tolist(),
                'frecuencias': df[col].value_counts().to_dict(),
                'tipo_codificacion': 'numérica' if pd.api.types.is_numeric_dtype(df[col]) else 'texto'
            }
    
    print("\n🔘 VARIABLES BINARIAS DETECTADAS")
    print("=" * 80)
    for col, info in binarias.items():
        print(f"\n📌 {col}")
        print(f"   Valores: {info['valores']}")
        print(f"   Frecuencias: {info['frecuencias']}")
        print(f"   Tipo: {info['tipo_codificacion']}")
    
    return binarias

binarias = detectar_binarias(df)
```

#### 🆔 Variables Identificadoras

```python
def detectar_identificadores(df, umbral=0.95):
    """
    Detecta columnas que podrían ser identificadores únicos
    """
    identificadores = []
    
    for col in df.columns:
        prop_unicos = df[col].nunique() / len(df)
        
        if prop_unicos > umbral:
            identificadores.append({
                'columna': col,
                'valores_unicos': df[col].nunique(),
                'total_registros': len(df),
                'proporcion_unica': f"{prop_unicos * 100:.2f}%",
                'tipo': 'ID probable'
            })
    
    if identificadores:
        print("\n🆔 IDENTIFICADORES DETECTADOS")
        print("=" * 80)
        df_ids = pd.DataFrame(identificadores)
        print(df_ids.to_string(index=False))
        print("\n⚠️  Estas columnas probablemente no deben usarse en el modelado")
    else:
        print("\n✅ No se detectaron identificadores obvios")
    
    return identificadores

ids = detectar_identificadores(df)
```

#### 📅 Variables de Fecha/Tiempo

```python
def detectar_fechas(df):
    """
    Detecta columnas que podrían contener fechas
    """
    fechas = []
    
    for col in df.columns:
        # Si ya es datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            fechas.append({
                'columna': col,
                'tipo': 'datetime',
                'min': df[col].min(),
                'max': df[col].max(),
                'rango_dias': (df[col].max() - df[col].min()).days
            })
        
        # Intentar convertir strings a fechas
        elif pd.api.types.is_string_dtype(df[col]):
            try:
                fecha_test = pd.to_datetime(df[col].dropna().iloc[:100], errors='coerce')
                if fecha_test.notna().sum() > 50:  # Al menos 50% son fechas válidas
                    fechas.append({
                        'columna': col,
                        'tipo': 'string (convertible a fecha)',
                        'formato_detectado': 'automático',
                        'conversion_sugerida': f"pd.to_datetime(df['{col}'])"
                    })
            except:
                pass
    
    if fechas:
        print("\n📅 VARIABLES DE FECHA/TIEMPO")
        print("=" * 80)
        df_fechas = pd.DataFrame(fechas)
        print(df_fechas.to_string(index=False))
    
    return fechas

fechas = detectar_fechas(df)
```

---

### 0.4 Reporte Completo Automatizado

```python
def generar_reporte_completo(df, nombre_archivo='reporte_variables.txt'):
    """
    Genera un reporte completo en texto con toda la información
    """
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" REPORTE COMPLETO DE VARIABLES ".center(100, "=") + "\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nDataset: {df.shape[0]} filas × {df.shape[1]} columnas\n")
        f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Clasificación
        f.write("\n" + "-" * 100 + "\n")
        f.write("CLASIFICACIÓN DE VARIABLES\n")
        f.write("-" * 100 + "\n")
        
        clasificacion = clasificar_variables(df)
        for tipo, columnas in clasificacion.items():
            if columnas:
                f.write(f"\n{tipo.upper().replace('_', ' ')} ({len(columnas)}):\n")
                for col in columnas:
                    f.write(f"  • {col}\n")
        
        # Resumen detallado
        f.write("\n" + "-" * 100 + "\n")
        f.write("RESUMEN DETALLADO POR COLUMNA\n")
        f.write("-" * 100 + "\n\n")
        
        df_resumen = generar_resumen_columnas(df)
        f.write(df_resumen.to_string(index=False))
        
        # Recomendaciones
        f.write("\n\n" + "-" * 100 + "\n")
        f.write("RECOMENDACIONES\n")
        f.write("-" * 100 + "\n")
        
        # Detectar problemas
        nulos_altos = df_resumen[df_resumen['Valores_Nulos'] > len(df) * 0.5]
        if len(nulos_altos) > 0:
            f.write(f"\n⚠️  Variables con >50% de valores nulos ({len(nulos_altos)}):\n")
            for col in nulos_altos['Columna']:
                f.write(f"   • {col}\n")
            f.write("   → Considerar eliminar o imputar\n")
        
        # Identificadores
        ids = detectar_identificadores(df)
        if ids:
            f.write(f"\n🆔 Variables identificadoras detectadas ({len(ids)}):\n")
            for id_info in ids:
                f.write(f"   • {id_info['columna']}\n")
            f.write("   → No usar en modelado predictivo\n")
    
    print(f"\n✅ Reporte generado: {nombre_archivo}")
    return nombre_archivo

# Generar reporte
reporte = generar_reporte_completo(df, 'reporte_variables.txt')
```

---

### 0.5 Pipeline Completo de Identificación

```python
def pipeline_identificacion_variables(ruta_csv):
    """
    Pipeline completo para identificar y analizar tipos de variables
    """
    print("🚀 Iniciando pipeline de identificación de variables...")
    print("=" * 100)
    
    # 1. Cargar datos
    print("\n📂 Paso 1: Cargando datos...")
    df = pd.read_csv(ruta_csv)
    print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # 2. Inspección básica
    print("\n🔍 Paso 2: Inspección básica...")
    inspeccionar_dataset(df)
    
    # 3. Clasificación automática
    print("\n🧠 Paso 3: Clasificación automática...")
    clasificacion = clasificar_variables(df)
    mostrar_clasificacion(clasificacion, df)
    
    # 4. Detección de variables especiales
    print("\n🔎 Paso 4: Detectando variables especiales...")
    binarias = detectar_binarias(df)
    ids = detectar_identificadores(df)
    fechas = detectar_fechas(df)
    
    # 5. Generar resumen de columnas
    print("\n📊 Paso 5: Generando resumen de columnas...")
    df_resumen = generar_resumen_columnas(df)
    df_resumen.to_csv('resumen_columnas.csv', index=False)
    print("✅ Resumen exportado a 'resumen_columnas.csv'")
    
    # 6. Visualizaciones
    print("\n🎨 Paso 6: Generando visualizaciones...")
    visualizar_tipos_variables(clasificacion, df)
    
    # 7. Reporte completo
    print("\n📝 Paso 7: Generando reporte completo...")
    generar_reporte_completo(df, 'reporte_variables.txt')
    
    print("\n" + "=" * 100)
    print("✅ Pipeline completado exitosamente!")
    print("=" * 100)
    
    return df, clasificacion, df_resumen

# Ejecutar pipeline completo
df, clasificacion, resumen = pipeline_identificacion_variables('framingham.csv')
```

---

### 0.6 Guía de Análisis por Tipo de Variable

Una vez identificados los tipos de variables, es crucial saber **qué análisis aplicar a cada una**. Esta sección detalla las técnicas estadísticas, visualizaciones y métodos de detección de outliers apropiados para cada tipo.

---

#### 📊 TABLA RESUMEN: Análisis por Tipo de Variable

| Tipo Variable | Estadísticos | Visualizaciones | Detección Outliers | Pruebas Estadísticas |
|---------------|--------------|-----------------|-------------------|----------------------|
| **Continua** | Media, Mediana, DE, CV, Asimetría, Curtosis | Histograma, Boxplot, Densidad, QQ-Plot | ✅ Tukey (IQR)<br>✅ Z-Score<br>✅ Isolation Forest | t-test, ANOVA, Pearson |
| **Discreta (muchos valores)** | Media, Mediana, Moda, Rango | Histograma, Barras, Boxplot | ✅ Tukey (IQR)<br>⚠️ Contexto | t-test, ANOVA, Chi² |
| **Discreta (pocos valores)** | Moda, Frecuencias | Barras, Pastel | ❌ Análisis de frecuencias | Chi-Cuadrado, Cochran |
| **Binaria** | Proporción, Odds Ratio | Barras, Pastel | ❌ No aplica | Chi², Fisher, McNemar |
| **Nominal** | Moda, Frecuencias, Entropía | Barras, Pastel, Heatmap | ❌ No aplica | Chi², Cramér's V |
| **Ordinal** | Mediana, Moda, Percentiles | Barras ordenadas | ❌ Análisis contextual | Mann-Whitney, Kruskal-Wallis |
| **Fecha/Tiempo** | Rango, Tendencias, Estacionalidad | Series temporales, Líneas | ✅ Fechas imposibles | Pruebas de tendencia |
| **Identificador** | Conteo de únicos | ❌ No aplica | ✅ Duplicados | ❌ Excluir del análisis |

---

#### 🔬 Análisis Detallado por Tipo

##### 1️⃣ **Variables Numéricas Continuas**

**Ejemplos del dataset Framingham:** `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`

```python
def analisis_completo_continua(df, columna):
    """Análisis completo para variable continua"""
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISIS: {columna} (VARIABLE CONTINUA)")
    print(f"{'='*80}")
    
    datos = df[columna].dropna()
    
    # 1. ESTADÍSTICOS DESCRIPTIVOS
    print("\n1️⃣ ESTADÍSTICOS DESCRIPTIVOS")
    print(f"   N: {len(datos)}")
    print(f"   Media: {datos.mean():.2f}")
    print(f"   Mediana: {datos.median():.2f}")
    print(f"   Desviación Estándar: {datos.std():.2f}")
    print(f"   Coeficiente de Variación: {(datos.std()/datos.mean()*100):.2f}%")
    print(f"   Asimetría: {datos.skew():.2f}")
    print(f"   Curtosis: {datos.kurtosis():.2f}")
    print(f"   Rango: [{datos.min():.2f}, {datos.max():.2f}]")
    
    # 2. DETECCIÓN DE OUTLIERS (Tukey)
    print("\n2️⃣ DETECCIÓN DE OUTLIERS (Método Tukey)")
    Q1 = datos.quantile(0.25)
    Q3 = datos.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < lower) | (df[columna] > upper)]
    
    print(f"   Q1 (25%): {Q1:.2f}")
    print(f"   Q3 (75%): {Q3:.2f}")
    print(f"   IQR: {IQR:.2f}")
    print(f"   Límite inferior: {lower:.2f}")
    print(f"   Límite superior: {upper:.2f}")
    print(f"   Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    # 3. PRUEBA DE NORMALIDAD
    print("\n3️⃣ PRUEBA DE NORMALIDAD (Shapiro-Wilk)")
    from scipy import stats
    if len(datos) < 5000:  # Shapiro funciona mejor con n < 5000
        stat, p_value = stats.shapiro(datos.sample(min(5000, len(datos))))
        print(f"   Estadístico: {stat:.4f}")
        print(f"   P-valor: {p_value:.4f}")
        print(f"   Conclusión: {'Distribución NORMAL' if p_value > 0.05 else 'Distribución NO NORMAL'}")
    
    # 4. VISUALIZACIONES
    print("\n4️⃣ VISUALIZACIONES RECOMENDADAS")
    print("   ✅ Histograma + Curva de densidad")
    print("   ✅ Boxplot (para ver outliers)")
    print("   ✅ QQ-Plot (evaluar normalidad)")
    print("   ✅ Violin Plot")
    
    # 5. ANÁLISIS BIVARIADO POSIBLE
    print("\n5️⃣ ANÁLISIS BIVARIADO")
    print("   • Con otra continua → Correlación de Pearson + Scatter plot")
    print("   • Con categórica → ANOVA / t-test + Boxplots por grupo")

# Ejemplo: Análisis de glucosa
analisis_completo_continua(df, 'glucose')
```

**Visualización recomendada:**
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma + Densidad
sns.histplot(df['glucose'], kde=True, bins=30, ax=axes[0,0])
axes[0,0].axvline(df['glucose'].mean(), color='red', linestyle='--', label='Media')
axes[0,0].axvline(df['glucose'].median(), color='green', linestyle='--', label='Mediana')
axes[0,0].legend()
axes[0,0].set_title('Distribución de Glucosa')

# Boxplot
sns.boxplot(x=df['glucose'], ax=axes[0,1])
axes[0,1].set_title('Boxplot - Detección de Outliers')

# QQ-Plot
from scipy import stats
stats.probplot(df['glucose'].dropna(), dist="norm", plot=axes[1,0])
axes[1,0].set_title('QQ-Plot - Normalidad')

# Violin Plot
sns.violinplot(y=df['glucose'], ax=axes[1,1])
axes[1,1].set_title('Violin Plot')

plt.tight_layout()
plt.show()
```

---

##### 2️⃣ **Variables Numéricas Discretas (Muchos Valores)**

**Ejemplos del dataset Framingham:** `age` (39 valores), `cigsPerDay` (33 valores)

```python
def analisis_completo_discreta(df, columna):
    """Análisis para variable discreta con muchos valores"""
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISIS: {columna} (VARIABLE DISCRETA)")
    print(f"{'='*80}")
    
    datos = df[columna].dropna()
    n_unicos = datos.nunique()
    
    # 1. ESTADÍSTICOS
    print("\n1️⃣ ESTADÍSTICOS DESCRIPTIVOS")
    print(f"   N: {len(datos)}")
    print(f"   Valores únicos: {n_unicos}")
    print(f"   Media: {datos.mean():.2f}")
    print(f"   Mediana: {datos.median():.2f}")
    print(f"   Moda: {datos.mode().values[0]}")
    print(f"   Desviación Estándar: {datos.std():.2f}")
    print(f"   Rango: [{int(datos.min())}, {int(datos.max())}]")
    
    # 2. FRECUENCIAS (Top 10)
    print("\n2️⃣ TABLA DE FRECUENCIAS (Top 10)")
    freq = datos.value_counts().head(10)
    for valor, count in freq.items():
        print(f"   {valor}: {count} ({count/len(datos)*100:.2f}%)")
    
    # 3. OUTLIERS (solo si tiene muchos valores)
    if n_unicos > 20:
        print("\n3️⃣ DETECCIÓN DE OUTLIERS (Método Tukey)")
        Q1 = datos.quantile(0.25)
        Q3 = datos.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[columna] < lower) | (df[columna] > upper)]
        print(f"   Límites: [{lower:.2f}, {upper:.2f}]")
        print(f"   Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    else:
        print("\n3️⃣ DETECCIÓN DE OUTLIERS")
        print("   ⚠️ Pocos valores únicos - Se recomienda análisis contextual")
    
    # 4. VISUALIZACIONES
    print("\n4️⃣ VISUALIZACIONES RECOMENDADAS")
    print("   ✅ Histograma (con bins discretos)")
    print("   ✅ Gráfico de barras")
    print("   ✅ Boxplot (si n_unicos > 20)")
    
    # 5. ANÁLISIS BIVARIADO
    print("\n5️⃣ ANÁLISIS BIVARIADO")
    print("   • Con otra numérica → Correlación de Spearman")
    print("   • Con categórica → ANOVA / Chi² + Gráficos por grupo")

# Ejemplo: Análisis de edad
analisis_completo_discreta(df, 'age')
```

---

##### 3️⃣ **Variables Numéricas Discretas (Pocos Valores)**

**Ejemplos del dataset Framingham:** `education` (4 valores: 1, 2, 3, 4)

```python
def analisis_discreta_pocos_valores(df, columna):
    """Análisis para variable discreta con pocos valores (ordinal)"""
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISIS: {columna} (VARIABLE ORDINAL)")
    print(f"{'='*80}")
    
    datos = df[columna].dropna()
    
    # 1. FRECUENCIAS COMPLETAS
    print("\n1️⃣ DISTRIBUCIÓN DE FRECUENCIAS")
    freq_abs = datos.value_counts().sort_index()
    freq_rel = datos.value_counts(normalize=True).sort_index() * 100
    
    for valor in sorted(datos.unique()):
        abs_count = freq_abs.get(valor, 0)
        rel_count = freq_rel.get(valor, 0)
        print(f"   Nivel {int(valor)}: {abs_count:4d} ({rel_count:5.2f}%)")
    
    # 2. ESTADÍSTICOS APROPIADOS
    print("\n2️⃣ ESTADÍSTICOS (Para datos ordinales)")
    print(f"   N: {len(datos)}")
    print(f"   Moda: {datos.mode().values[0]}")
    print(f"   Mediana: {datos.median()}")
    print(f"   Q1 (25%): {datos.quantile(0.25)}")
    print(f"   Q3 (75%): {datos.quantile(0.75)}")
    
    # 3. NO OUTLIERS
    print("\n3️⃣ DETECCIÓN DE OUTLIERS")
    print("   ❌ NO APLICA - Variable ordinal con pocos niveles")
    print("   → Todos los valores son categorías válidas")
    
    # 4. VISUALIZACIONES
    print("\n4️⃣ VISUALIZACIONES RECOMENDADAS")
    print("   ✅ Gráfico de barras (ordenado)")
    print("   ✅ Gráfico de pastel")
    print("   ❌ Histograma continuo (NO recomendado)")
    print("   ❌ Boxplot (NO recomendado)")
    
    # 5. ANÁLISIS BIVARIADO
    print("\n5️⃣ ANÁLISIS BIVARIADO")
    print("   • Con otra ordinal → Correlación de Spearman")
    print("   • Con categórica → Chi-Cuadrado")
    print("   • Con continua → Kruskal-Wallis / Comparación por niveles")

# Ejemplo: Análisis de educación
analisis_discreta_pocos_valores(df, 'education')
```

---

##### 4️⃣ **Variables Binarias**

**Ejemplos del dataset Framingham:** `male`, `currentSmoker`, `BPMeds`, `diabetes`, `TenYearCHD`

```python
def analisis_binaria(df, columna):
    """Análisis completo para variable binaria"""
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISIS: {columna} (VARIABLE BINARIA)")
    print(f"{'='*80}")
    
    datos = df[columna].dropna()
    valores = sorted(datos.unique())
    
    # 1. DISTRIBUCIÓN
    print("\n1️⃣ DISTRIBUCIÓN")
    freq = datos.value_counts()
    for valor in valores:
        count = freq.get(valor, 0)
        pct = count / len(datos) * 100
        etiqueta = "No" if valor == 0 else "Sí"
        print(f"   {etiqueta} ({valor}): {count:4d} ({pct:5.2f}%)")
    
    # 2. ESTADÍSTICOS APROPIADOS
    print("\n2️⃣ ESTADÍSTICOS")
    prop_1 = datos.sum() / len(datos)
    prop_0 = 1 - prop_1
    print(f"   Proporción (1): {prop_1:.4f} ({prop_1*100:.2f}%)")
    print(f"   Proporción (0): {prop_0:.4f} ({prop_0*100:.2f}%)")
    print(f"   Odds Ratio: {prop_1/prop_0:.4f}")
    
    # 3. DESBALANCE
    print("\n3️⃣ ANÁLISIS DE DESBALANCE")
    ratio = min(freq) / max(freq)
    if ratio < 0.1:
        nivel = "CRÍTICO"
    elif ratio < 0.3:
        nivel = "ALTO"
    elif ratio < 0.5:
        nivel = "MODERADO"
    else:
        nivel = "BALANCEADO"
    print(f"   Ratio: {ratio:.4f}")
    print(f"   Nivel de desbalance: {nivel}")
    if ratio < 0.3:
        print("   ⚠️ Considerar técnicas de balanceo para modelado")
    
    # 4. NO OUTLIERS
    print("\n4️⃣ DETECCIÓN DE OUTLIERS")
    print("   ❌ NO APLICA - Variable binaria")
    
    # 5. VISUALIZACIONES
    print("\n5️⃣ VISUALIZACIONES RECOMENDADAS")
    print("   ✅ Gráfico de barras")
    print("   ✅ Gráfico de pastel")
    print("   ❌ Histograma (NO recomendado)")
    print("   ❌ Boxplot (NO recomendado)")
    
    # 6. ANÁLISIS BIVARIADO
    print("\n6️⃣ ANÁLISIS BIVARIADO")
    print("   • Con otra binaria → Chi-Cuadrado, Odds Ratio, Test de Fisher")
    print("   • Con categórica → Chi-Cuadrado, Cramér's V")
    print("   • Con continua → t-test, Mann-Whitney, Boxplot por grupo")
    
    # 7. INTERVALO DE CONFIANZA
    from scipy import stats
    n = len(datos)
    p = prop_1
    se = np.sqrt(p * (1-p) / n)
    ci_lower = p - 1.96 * se
    ci_upper = p + 1.96 * se
    print("\n7️⃣ INTERVALO DE CONFIANZA (95%)")
    print(f"   Proporción: {p:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# Ejemplo: Análisis de diabetes
analisis_binaria(df, 'diabetes')
```

**Visualización recomendada:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Gráfico de barras
freq = df['diabetes'].value_counts()
axes[0].bar(['No (0)', 'Sí (1)'], freq.values, color=['lightblue', 'salmon'])
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Diabetes')
for i, v in enumerate(freq.values):
    axes[0].text(i, v + 10, str(v), ha='center')

# Gráfico de pastel
axes[1].pie(freq.values, labels=['No', 'Sí'], autopct='%1.1f%%', 
            colors=['lightblue', 'salmon'])
axes[1].set_title('Proporción de Diabetes')

# Comparación con variable continua
sns.boxplot(x='diabetes', y='glucose', data=df, ax=axes[2])
axes[2].set_xticklabels(['No Diabetes', 'Diabetes'])
axes[2].set_title('Glucosa por Estado de Diabetes')

plt.tight_layout()
plt.show()
```

---

#### 🎯 Matriz de Decisión: ¿Qué Análisis Usar?

```python
def recomendar_analisis(df, columna):
    """
    Recomienda automáticamente los análisis apropiados para una variable
    """
    n_unicos = df[columna].nunique()
    dtype = df[columna].dtype
    
    print(f"\n{'='*80}")
    print(f"🎯 RECOMENDACIONES PARA: {columna}")
    print(f"{'='*80}")
    print(f"Tipo de dato: {dtype}")
    print(f"Valores únicos: {n_unicos}")
    
    # CLASIFICACIÓN
    if n_unicos == 1:
        tipo = "CONSTANTE"
        print(f"\n⚠️ Variable CONSTANTE - Eliminar del análisis")
        return
    
    elif n_unicos == 2:
        tipo = "BINARIA"
        print(f"\nTipo: {tipo}")
        print("\n📊 ANÁLISIS RECOMENDADOS:")
        print("  ✅ Tabla de frecuencias y proporciones")
        print("  ✅ Gráfico de barras / pastel")
        print("  ✅ Odds Ratio")
        print("  ✅ Intervalo de confianza")
        print("\n🚫 NO USAR:")
        print("  ❌ Media / Desviación estándar")
        print("  ❌ Detección de outliers")
        print("  ❌ Histograma / Boxplot")
        print("\n🔗 ANÁLISIS BIVARIADO:")
        print("  • Con binaria → Chi², Fisher, Odds Ratio")
        print("  • Con continua → t-test, Boxplot por grupo")
    
    elif n_unicos < 10:
        tipo = "CATEGÓRICA / ORDINAL"
        print(f"\nTipo: {tipo}")
        print("\n📊 ANÁLISIS RECOMENDADOS:")
        print("  ✅ Tabla de frecuencias")
        print("  ✅ Gráfico de barras ordenado")
        print("  ✅ Moda y Mediana")
        print("  ✅ Percentiles")
        print("\n🚫 NO USAR:")
        print("  ❌ Media (si no tiene sentido matemático)")
        print("  ❌ Detección de outliers (valores atípicos)")
        print("  ❌ Histograma continuo")
        print("\n🔗 ANÁLISIS BIVARIADO:")
        print("  • Con categórica → Chi-Cuadrado")
        print("  • Con continua → ANOVA / Kruskal-Wallis")
    
    elif pd.api.types.is_numeric_dtype(df[columna]):
        if n_unicos < 50:
            tipo = "DISCRETA"
            print(f"\nTipo: {tipo}")
            print("\n📊 ANÁLISIS RECOMENDADOS:")
            print("  ✅ Media, Mediana, Moda, Desviación Estándar")
            print("  ✅ Histograma con bins discretos")
            print("  ✅ Gráfico de barras")
            print("  ✅ Detección de outliers (Tukey)")
            print("  ✅ Boxplot")
        else:
            tipo = "CONTINUA"
            print(f"\nTipo: {tipo}")
            print("\n📊 ANÁLISIS RECOMENDADOS:")
            print("  ✅ Todos los estadísticos descriptivos")
            print("  ✅ Histograma + Curva de densidad")
            print("  ✅ Boxplot")
            print("  ✅ QQ-Plot (normalidad)")
            print("  ✅ Detección de outliers (Tukey, Z-Score)")
        
        print("\n🔗 ANÁLISIS BIVARIADO:")
        print("  • Con numérica → Correlación Pearson/Spearman, Scatter plot")
        print("  • Con categórica → ANOVA / t-test, Boxplot por grupo")
    
    else:
        tipo = "TEXTO / OTRO"
        print(f"\nTipo: {tipo}")
        print("\n⚠️ Requiere análisis específico (NLP, conversión, etc.)")

# Aplicar a todas las variables
print("\n" + "="*100)
print(" RECOMENDACIONES DE ANÁLISIS PARA TODAS LAS VARIABLES ".center(100, "="))
print("="*100)

for col in df.columns:
    recomendar_analisis(df, col)
    print("\n")
```

---

#### 📝 Resumen Ejecutivo para Dataset Framingham

```python
def resumen_analisis_framingham(df):
    """
    Resumen específico de qué análisis hacer con el dataset Framingham
    """
    print("\n" + "="*100)
    print(" GUÍA DE ANÁLISIS - DATASET FRAMINGHAM ".center(100, "="))
    print("="*100)
    
    analisis_plan = {
        'CONTINUAS (Análisis Completo)': {
            'variables': ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'],
            'acciones': [
                '✅ Estadísticos: Media, Mediana, DE, CV, Asimetría, Curtosis',
                '✅ Outliers: Método Tukey (IQR)',
                '✅ Normalidad: Shapiro-Wilk, QQ-Plot',
                '✅ Visuales: Histograma, Boxplot, Violin, Densidad',
                '✅ Bivariado: Correlación Pearson + Scatter plots'
            ]
        },
        'DISCRETAS - Muchos valores (age, cigsPerDay)': {
            'variables': ['age', 'cigsPerDay'],
            'acciones': [
                '✅ Estadísticos: Media, Mediana, Moda, DE',
                '✅ Outliers: Método Tukey (aplicable)',
                '✅ Visuales: Histograma discreto, Barras',
                '✅ Bivariado: Correlación Spearman'
            ]
        },
        'ORDINAL - Pocos valores (education)': {
            'variables': ['education'],
            'acciones': [
                '✅ Estadísticos: Mediana, Moda, Percentiles',
                '❌ NO outliers estadísticos',
                '✅ Visuales: Barras ordenadas',
                '✅ Bivariado: Chi² con categóricas, Kruskal-Wallis con continuas'
            ]
        },
        'BINARIAS (7 variables)': {
            'variables': ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 
                         'prevalentHyp', 'diabetes', 'TenYearCHD'],
            'acciones': [
                '✅ Proporciones e Intervalos de Confianza',
                '✅ Análisis de desbalance',
                '❌ NO Media, NO Outliers',
                '✅ Visuales: Barras, Pastel',
                '✅ Bivariado: Chi² entre binarias, t-test con continuas'
            ]
        }
    }
    
    for categoria, info in analisis_plan.items():
        print(f"\n{'='*100}")
        print(f"📊 {categoria}")
        print(f"{'='*100}")
        print(f"\nVariables: {', '.join(info['variables'])}")
        print("\nAcciones recomendadas:")
        for accion in info['acciones']:
            print(f"  {accion}")
    
    # ANÁLISIS BIVARIADOS CLAVE
    print(f"\n{'='*100}")
    print("🔗 ANÁLISIS BIVARIADOS PRIORITARIOS")
    print(f"{'='*100}")
    
    bivariados = [
        ("TenYearCHD (objetivo)", "Todas las demás", "Variable objetivo - Analizar relación con todos los predictores"),
        ("glucose", "diabetes", "t-test / Boxplot - Verificar diferencias"),
        ("sysBP", "diaBP", "Correlación Pearson - Esperada alta correlación"),
        ("age", "TenYearCHD", "Chi² / Regresión logística - Factor de riesgo clave"),
        ("currentSmoker", "TenYearCHD", "Chi² / Odds Ratio - Factor de riesgo"),
        ("BMI", "diabetes", "t-test / Boxplot - Relación esperada"),
    ]
    
    for var1, var2, descripcion in bivariados:
        print(f"\n  • {var1} vs {var2}")
        print(f"    → {descripcion}")

resumen_analisis_framingham(df)
```

---

## 1. Perfilado de Calidad y Limpieza

El primer paso es asegurar la integridad de la "Matriz de Datos". La calidad no es solo ausencia de nulos, sino **consistencia lógica y representatividad**.

> **Regla de Oro**: El preprocesamiento ocupa entre el 60% y 90% del tiempo en un proyecto de ciencia de datos.

---

### 1.0 Tratamiento de Valores Nulos

Antes de cualquier análisis, debemos decidir qué hacer con los valores faltantes.

#### 📋 Estrategias según Tipo de Variable

```python
# DETECTAR VALORES NULOS
print("=== VALORES NULOS POR COLUMNA ===")
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
resumen_nulos = pd.DataFrame({
    'Nulos': nulos,
    'Porcentaje': nulos_pct
})
print(resumen_nulos[resumen_nulos['Nulos'] > 0].sort_values('Nulos', ascending=False))
```

#### 🔧 Opciones de Tratamiento

##### **Opción 1: Eliminar filas con nulos**
✅ **Usar cuando:** < 5% de filas afectadas  
❌ **No usar cuando:** Se pierde mucha información

```python
# Ejemplo simple: Eliminar filas con nulos en columnas específicas
df_limpio = df.dropna(subset=['glucose', 'BMI'])
print(f"Filas originales: {len(df)}")
print(f"Filas después de eliminar: {len(df_limpio)}")
print(f"Filas eliminadas: {len(df) - len(df_limpio)} ({(len(df)-len(df_limpio))/len(df)*100:.2f}%)")

# Eliminar solo si tiene muchos nulos
umbral = 0.5  # 50%
df_limpio = df.dropna(thresh=len(df.columns) * umbral)
```

##### **Opción 2: Imputar con estadísticos (Variables numéricas)**

```python
# A) IMPUTAR CON LA MEDIA (variables continuas sin outliers)
# ✅ Usar: glucose, totChol, heartRate
# ❌ No usar: Si hay muchos outliers
df['glucose'] = df['glucose'].fillna(df['glucose'].mean())

# B) IMPUTAR CON LA MEDIANA (variables con outliers)
# ✅ Usar: sysBP, diaBP (más robusto)
df['sysBP'] = df['sysBP'].fillna(df['sysBP'].median())

# C) IMPUTAR CON LA MODA (variables discretas)
# ✅ Usar: education, cigsPerDay
df['education'] = df['education'].fillna(df['education'].mode()[0])

# EJEMPLO COMPLETO PARA FRAMINGHAM
def imputar_nulos_numericas(df):
    """Imputar valores nulos en variables numéricas"""
    
    # Variables continuas → mediana (más robusta)
    continuas = ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    for col in continuas:
        if df[col].isnull().sum() > 0:
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)
            print(f"✅ {col}: {df[col].isnull().sum()} nulos imputados con mediana = {mediana:.2f}")
    
    # Variables discretas → moda
    discretas = ['education', 'cigsPerDay']
    for col in discretas:
        if df[col].isnull().sum() > 0:
            moda = df[col].mode()[0]
            df[col].fillna(moda, inplace=True)
            print(f"✅ {col}: imputado con moda = {moda}")
    
    return df

df = imputar_nulos_numericas(df)
```

##### **Opción 3: Imputar con valores específicos (Variables categóricas/binarias)**

```python
# Para binarias: imputar con el valor más frecuente
df['BPMeds'] = df['BPMeds'].fillna(0)  # Asumimos "No" (0)

# Para categóricas: crear categoría "Desconocido"
df['education'] = df['education'].fillna('Desconocido')
```

##### **Opción 4: Imputación avanzada (Cuando hay muchos nulos)**

```python
# Usar KNN Imputer (imputa basándose en vecinos similares)
from sklearn.impute import KNNImputer

# Seleccionar solo columnas numéricas
numericas = df.select_dtypes(include=[np.number]).columns
df_num = df[numericas]

# Aplicar KNN
imputer = KNNImputer(n_neighbors=5)
df_imputado = pd.DataFrame(
    imputer.fit_transform(df_num),
    columns=numericas
)

# Reemplazar en el dataframe original
df[numericas] = df_imputado
print("✅ Imputación KNN completada")
```

##### **Opción 5: Crear columna indicadora**

```python
# Crear variable que indique si tenía nulo (útil para modelado)
df['glucose_faltante'] = df['glucose'].isnull().astype(int)
df['glucose'] = df['glucose'].fillna(df['glucose'].median())

print(f"Registros con glucose faltante: {df['glucose_faltante'].sum()}")
```

---

### 1.1 Calidad en Variables Categóricas (Nominales/Ordinales)

A diferencia de las variables numéricas, aquí los "valores malos" se identifican mediante:

#### 🔍 Técnicas de Detección

| Problema | Descripción | Ejemplo |
|----------|-------------|---------|
| **Nulos enmascarados** | Valores que representan "ausencia" pero no son `NaN` | `" "`, `"?"`, `"N/A"`, `"null"`, `"-"`, `"Unknown"` |
| **Inconsistencias de etiqueta** | Variaciones de escritura de la misma categoría | `"Masc"`, `"Masculino"`, `"M"`, `"MASCULINO"` |
| **Espacios extra** | Caracteres invisibles que dificultan el match | `"Mujer "` vs `"Mujer"`, `" Sí"` vs `"Sí"` |
| **Categorías de baja frecuencia** | Niveles con representatividad < 1-5% | Categoría con 3 registros de 10,000 |
| **Mayúsculas inconsistentes** | Diferentes casos de la misma palabra | `"activo"`, `"Activo"`, `"ACTIVO"` |

#### 🛠️ Estrategias de Limpieza

```python
import pandas as pd
import numpy as np

# 1. Normalizar strings (espacios, mayúsculas)
df['columna'] = df['columna'].str.strip().str.lower()

# 2. Reemplazar nulos enmascarados
valores_nulos = ['?', 'N/A', 'NULL', 'unknown', '-', ' ', 'nan', 'None']
df.replace(valores_nulos, np.nan, inplace=True)

# 3. Unificar categorías similares
diccionario_reemplazo = {
    'masc': 'masculino',
    'm': 'masculino',
    'fem': 'femenino',
    'f': 'femenino'
}
df['genero'] = df['genero'].replace(diccionario_reemplazo)

# 4. Agrupar categorías de baja frecuencia
def agrupar_raros(columna, umbral=0.05):
    """Agrupa categorías con frecuencia menor al umbral en 'Otros'"""
    freq = columna.value_counts(normalize=True)
    categorias_raras = freq[freq < umbral].index
    return columna.apply(lambda x: 'Otros' if x in categorias_raras else x)

df['pais'] = agrupar_raros(df['pais'], umbral=0.03)

# 5. Validación de rangos en ordinales
# Ej: Nivel educativo debe estar en ['Primaria', 'Secundaria', 'Universidad']
valores_validos = ['primaria', 'secundaria', 'universidad']
df.loc[~df['educacion'].isin(valores_validos), 'educacion'] = np.nan
```

#### 📊 Análisis de Frecuencias

```python
# Análisis completo de una variable categórica
def analizar_categorica(df, columna):
    print(f"=== Análisis de {columna} ===")
    print(f"\nValores únicos: {df[columna].nunique()}")
    print(f"Valores nulos: {df[columna].isna().sum()} ({df[columna].isna().mean()*100:.2f}%)")
    print("\nFrecuencias:")
    freq_abs = df[columna].value_counts(dropna=False)
    freq_rel = df[columna].value_counts(normalize=True, dropna=False) * 100
    tabla = pd.DataFrame({
        'Frecuencia': freq_abs,
        'Porcentaje': freq_rel.round(2)
    })
    print(tabla)
    
    # Identificar categorías de baja frecuencia
    raras = tabla[tabla['Porcentaje'] < 5]
    if len(raras) > 0:
        print(f"\n⚠️ Categorías con < 5%: {list(raras.index)}")
```

---

### 1.2 Calidad en Variables Cuantitativas (Valores Extremos)

Los **outliers** son observaciones que se alejan significativamente del comportamiento central y pueden ser:
- **Errores de medición o carga**: Edad = 250 años
- **Valores atípicos legítimos**: Salarios muy altos en una empresa
- **Casos especiales**: Ventas extraordinarias en fechas especiales

#### 📐 Método de Tukey (Rango Intercuartílico - IQR)

El método más robusto para detectar outliers:

$$Q1 = \text{Percentil 25}$$
$$Q3 = \text{Percentil 75}$$
$$IQR = Q3 - Q1$$

$$\text{Límite Inferior} = Q1 - 1.5 \times IQR$$
$$\text{Límite Superior} = Q3 + 1.5 \times IQR$$

**Interpretación:**
- Valores fuera de estos límites son considerados outliers moderados
- Para outliers extremos se usa $3 \times IQR$

```python
def detectar_outliers_tukey(df, columna, k=1.5):
    """
    Detecta outliers usando el método de Tukey
    k=1.5 para outliers moderados, k=3 para extremos
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - k * IQR
    limite_superior = Q3 + k * IQR
    
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    
    print(f"=== Outliers en {columna} ===")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Límites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
    print(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers, limite_inferior, limite_superior
```

#### 🎯 Método Z-Score (Distribución Normal)

Para datos que siguen una distribución normal:

$$Z = \frac{X - \mu}{\sigma}$$

**Criterio:** $|Z| > 3$ indica un outlier (valor a más de 3 desviaciones estándar de la media)

```python
from scipy import stats

def detectar_outliers_zscore(df, columna, umbral=3):
    """Detecta outliers usando Z-score (asume normalidad)"""
    z_scores = np.abs(stats.zscore(df[columna].dropna()))
    outliers_idx = np.where(z_scores > umbral)[0]
    
    print(f"Outliers detectados (|Z| > {umbral}): {len(outliers_idx)}")
    return df.iloc[outliers_idx]
```

#### 🔧 Estrategias de Tratamiento

```python
# 1. ELIMINACIÓN (usar con precaución)
df_limpio = df[(df['edad'] >= 0) & (df['edad'] <= 120)]

# 2. IMPUTACIÓN con límites
df['salario'] = df['salario'].clip(lower=limite_inferior, upper=limite_superior)

# 3. TRANSFORMACIÓN logarítmica (reduce el efecto de outliers)
df['salario_log'] = np.log1p(df['salario'])

# 4. MARCAR como categoría especial
df['outlier_salario'] = ((df['salario'] < limite_inferior) | 
                          (df['salario'] > limite_superior)).astype(int)

# 5. WINSORIZACIÓN (reemplazar extremos por percentiles)
from scipy.stats.mstats import winsorize
df['salario_wins'] = winsorize(df['salario'], limits=[0.05, 0.05])
```

#### 📊 Detección de Valores Imposibles

```python
def validar_rangos(df):
    """Valida que los valores estén en rangos lógicos"""
    problemas = []
    
    # Ejemplos de validaciones
    if 'edad' in df.columns:
        invalidos = df[(df['edad'] < 0) | (df['edad'] > 120)]
        if len(invalidos) > 0:
            problemas.append(f"Edad inválida: {len(invalidos)} registros")
    
    if 'temperatura' in df.columns:
        invalidos = df[(df['temperatura'] < -50) | (df['temperatura'] > 60)]
        if len(invalidos) > 0:
            problemas.append(f"Temperatura inválida: {len(invalidos)} registros")
    
    if 'porcentaje' in df.columns:
        invalidos = df[(df['porcentaje'] < 0) | (df['porcentaje'] > 100)]
        if len(invalidos) > 0:
            problemas.append(f"Porcentaje inválido: {len(invalidos)} registros")
    
    return problemas
```

---

#### 🔧 TRATAMIENTO DE OUTLIERS: Guía Práctica

Una vez detectados los outliers, debemos decidir qué hacer con ellos.

##### **Paso 1: Investigar el origen**

```python
# Ejemplo: Análisis de outliers en glucose
Q1 = df['glucose'].quantile(0.25)
Q3 = df['glucose'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

outliers = df[df['glucose'] > limite_superior]
print(f"Outliers en glucose: {len(outliers)}")
print("\nPrimeros 5 outliers:")
print(outliers[['glucose', 'diabetes', 'age']].head())
```

##### **Opción 1: NO hacer nada** ✅ Recomendado si:
- Son valores legítimos (ej: glucose alta en diabéticos)
- Representan casos importantes
- Son < 1% de los datos

```python
# Simplemente documentar
print(f"Se mantienen {len(outliers)} outliers por ser casos clínicos válidos")
```

##### **Opción 2: Eliminar outliers** ⚠️ Usar con precaución

```python
# Eliminar outliers extremos (k=3)
def eliminar_outliers(df, columna, k=3):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    df_limpio = df[(df[columna] >= lower) & (df[columna] <= upper)]
    eliminados = len(df) - len(df_limpio)
    
    print(f"Eliminadas {eliminados} filas ({eliminados/len(df)*100:.2f}%)")
    return df_limpio

# Ejemplo
df_sin_outliers = eliminar_outliers(df, 'glucose', k=3)
```

##### **Opción 3: Reemplazar con límites (Winsorización)** ✅ Más conservador

```python
# Reemplazar outliers por los límites del rango aceptable
def winsorizar(df, columna, k=1.5):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    # Reemplazar valores fuera de rango
    df[f'{columna}_winsorizado'] = df[columna].clip(lower=lower, upper=upper)
    
    reemplazados = ((df[columna] < lower) | (df[columna] > upper)).sum()
    print(f"✅ {reemplazados} valores ajustados a límites [{lower:.2f}, {upper:.2f}]")
    return df

# Ejemplo simple
df = winsorizar(df, 'glucose')
```

##### **Opción 4: Transformación logarítmica** ✅ Para distribuciones asimétricas

```python
# Reduce el efecto de outliers
df['glucose_log'] = np.log1p(df['glucose'])  # log1p = log(1 + x)

# Comparar distribuciones
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['glucose'].hist(bins=30, ax=axes[0])
axes[0].set_title('Glucose Original')
df['glucose_log'].hist(bins=30, ax=axes[1])
axes[1].set_title('Glucose Log-transformada')
plt.show()
```

##### **Opción 5: Imputar con estadísticos** ⚠️ Cambiar el outlier

```python
# Reemplazar outliers con la mediana
def imputar_outliers(df, columna, k=1.5):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    mediana = df[columna].median()
    
    # Marcar outliers
    mask_outliers = (df[columna] < lower) | (df[columna] > upper)
    
    # Reemplazar con mediana
    df.loc[mask_outliers, columna] = mediana
    
    print(f"✅ {mask_outliers.sum()} outliers reemplazados con mediana = {mediana:.2f}")
    return df

# Ejemplo
df = imputar_outliers(df.copy(), 'heartRate')
```

##### **Opción 6: Crear variable indicadora** ✅ Mantener la información

```python
# Crear columna que marque si era outlier (útil para modelos)
def marcar_outliers(df, columna, k=1.5):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    # Variable binaria: 1 si es outlier, 0 si no
    df[f'{columna}_es_outlier'] = ((df[columna] < lower) | (df[columna] > upper)).astype(int)
    
    print(f"✅ Variable indicadora creada: {df[f'{columna}_es_outlier'].sum()} outliers marcados")
    return df

# Ejemplo
df = marcar_outliers(df, 'totChol')
```

##### **🎯 Ejemplo Completo: Tratamiento de Outliers en Framingham**

```python
def tratar_outliers_framingham(df):
    """
    Pipeline de tratamiento de outliers para el dataset Framingham
    """
    print("=" * 80)
    print(" TRATAMIENTO DE OUTLIERS ".center(80, "="))
    print("=" * 80)
    
    # 1. glucose: Winsorizar (valores altos son clínicamente relevantes)
    print("\n1️⃣ GLUCOSE - Winsorización")
    df = winsorizar(df, 'glucose', k=1.5)
    
    # 2. totChol: Marcar outliers sin eliminar
    print("\n2️⃣ TOTAL CHOLESTEROL - Marcar outliers")
    df = marcar_outliers(df, 'totChol', k=1.5)
    
    # 3. heartRate: Imputar extremos con mediana
    print("\n3️⃣ HEART RATE - Imputar outliers extremos")
    Q1 = df['heartRate'].quantile(0.25)
    Q3 = df['heartRate'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR  # Usar k=3 para extremos
    upper = Q3 + 3 * IQR
    
    mask = (df['heartRate'] < lower) | (df['heartRate'] > upper)
    if mask.sum() > 0:
        df.loc[mask, 'heartRate'] = df['heartRate'].median()
        print(f"✅ {mask.sum()} outliers extremos imputados")
    
    # 4. BMI: Validar rangos lógicos
    print("\n4️⃣ BMI - Validación de rangos")
    invalidos = df[(df['BMI'] < 10) | (df['BMI'] > 60)]
    if len(invalidos) > 0:
        print(f"⚠️ {len(invalidos)} valores de BMI fuera de rango lógico")
        df.loc[invalidos.index, 'BMI'] = np.nan
        df['BMI'].fillna(df['BMI'].median(), inplace=True)
        print("✅ Valores inválidos reemplazados con mediana")
    
    print("\n" + "=" * 80)
    print("✅ Tratamiento de outliers completado")
    return df

# Aplicar
df = tratar_outliers_framingham(df)
```

---

## 2. Análisis Univariado

Objetivo: **Conocer la forma, centro y dispersión de cada variable individualmente**.

### 2.1 Variables Cuantitativas

#### 📈 Medidas de Tendencia Central

| Medida | Fórmula | Cuándo usar | Ventajas | Desventajas |
|--------|---------|-------------|----------|-------------|
| **Media** | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | Datos simétricos sin outliers | Usa todos los datos | Sensible a outliers |
| **Mediana** | Valor central ordenado | Datos con outliers o asimétricos | Robusta a outliers | Ignora magnitud de valores |
| **Moda** | Valor más frecuente | Datos discretos o categóricos | Identifica valor típico | Puede haber múltiples modas |

#### 📊 Medidas de Dispersión

| Medida | Fórmula | Interpretación |
|--------|---------|----------------|
| **Varianza** | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | Variabilidad promedio al cuadrado |
| **Desviación Estándar** | $s = \sqrt{s^2}$ | Dispersión en las mismas unidades que los datos |
| **Coeficiente de Variación** | $CV = \frac{s}{\bar{x}} \times 100$ | Dispersión relativa (%) - útil para comparar variables |
| **Rango** | $\text{Máx} - \text{Mín}$ | Amplitud total |
| **IQR** | $Q3 - Q1$ | Rango del 50% central (robusto) |

**Interpretación del CV:**
- CV < 15%: Baja dispersión (datos homogéneos)
- 15% ≤ CV ≤ 30%: Dispersión moderada
- CV > 30%: Alta dispersión (datos heterogéneos)

```python
def analisis_univariado_cuanti(df, columna):
    """Análisis estadístico completo de una variable cuantitativa"""
    datos = df[columna].dropna()
    
    print(f"=== Análisis de {columna} ===\n")
    
    # Tendencia central
    print("📊 TENDENCIA CENTRAL")
    print(f"Media: {datos.mean():.2f}")
    print(f"Mediana: {datos.median():.2f}")
    print(f"Moda: {datos.mode().values[0] if len(datos.mode()) > 0 else 'N/A'}")
    
    # Dispersión
    print("\n📈 DISPERSIÓN")
    print(f"Desviación Estándar: {datos.std():.2f}")
    print(f"Varianza: {datos.var():.2f}")
    print(f"Coeficiente de Variación: {(datos.std()/datos.mean()*100):.2f}%")
    print(f"Rango: [{datos.min():.2f}, {datos.max():.2f}]")
    print(f"IQR: {datos.quantile(0.75) - datos.quantile(0.25):.2f}")
    
    # Forma
    print("\n📐 FORMA DE LA DISTRIBUCIÓN")
    print(f"Asimetría (Skewness): {datos.skew():.2f}")
    print(f"  → Interpretación: {'Asimétrica izquierda' if datos.skew() < -0.5 else 'Simétrica' if abs(datos.skew()) <= 0.5 else 'Asimétrica derecha'}")
    print(f"Curtosis: {datos.kurtosis():.2f}")
    print(f"  → Interpretación: {'Platicúrtica (plana)' if datos.kurtosis() < -0.5 else 'Mesocúrtica (normal)' if abs(datos.kurtosis()) <= 0.5 else 'Leptocúrtica (puntiaguda)'}")
    
    # Cuartiles
    print("\n📦 CUARTILES")
    print(f"Q1 (25%): {datos.quantile(0.25):.2f}")
    print(f"Q2 (50%): {datos.quantile(0.50):.2f}")
    print(f"Q3 (75%): {datos.quantile(0.75):.2f}")
```

#### 📊 Visualizaciones Clave

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_univariado_cuanti(df, columna):
    """Visualizaciones completas para variable cuantitativa"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histograma con curva de densidad
    sns.histplot(df[columna], kde=True, bins=30, ax=axes[0, 0])
    axes[0, 0].set_title(f'Histograma + Densidad: {columna}')
    axes[0, 0].axvline(df[columna].mean(), color='red', linestyle='--', label='Media')
    axes[0, 0].axvline(df[columna].median(), color='green', linestyle='--', label='Mediana')
    axes[0, 0].legend()
    
    # 2. Boxplot
    sns.boxplot(x=df[columna], ax=axes[0, 1])
    axes[0, 1].set_title(f'Boxplot: {columna}')
    
    # 3. QQ-Plot (evaluar normalidad)
    from scipy import stats
    stats.probplot(df[columna].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q Plot: {columna}')
    
    # 4. Violin Plot (combina boxplot y densidad)
    sns.violinplot(y=df[columna], ax=axes[1, 1])
    axes[1, 1].set_title(f'Violin Plot: {columna}')
    
    plt.tight_layout()
    plt.show()
```

---

#### 🔧 TRATAMIENTO SEGÚN RESULTADOS DEL ANÁLISIS UNIVARIADO

##### **Según la Asimetría (Skewness)**

```python
# Calcular asimetría
skew = df['glucose'].skew()
print(f"Asimetría: {skew:.2f}")

# DECISIÓN:
if abs(skew) < 0.5:
    print("✅ Distribución simétrica - No requiere transformación")
elif skew > 0.5:
    print("⚠️ Asimetría positiva (cola derecha) - Aplicar transformación")
    
    # OPCIÓN 1: Transformación logarítmica
    df['glucose_log'] = np.log1p(df['glucose'])
    
    # OPCIÓN 2: Transformación raíz cuadrada
    df['glucose_sqrt'] = np.sqrt(df['glucose'])
    
    # OPCIÓN 3: Box-Cox (encuentra la mejor transformación)
    from scipy.stats import boxcox
    df['glucose_boxcox'], lambda_param = boxcox(df['glucose'] + 1)
    print(f"✅ Transformación aplicada (lambda = {lambda_param:.2f})")
    
elif skew < -0.5:
    print("⚠️ Asimetría negativa (cola izquierda)")
    # Transformación cuadrática
    df['variable_cuadrado'] = df['variable'] ** 2
```

##### **Según el Coeficiente de Variación (CV)**

```python
# Calcular CV
media = df['glucose'].mean()
std = df['glucose'].std()
cv = (std / media) * 100

print(f"Coeficiente de Variación: {cv:.2f}%")

# DECISIÓN:
if cv < 15:
    print("✅ Datos homogéneos - No requiere estandarización para análisis")
elif 15 <= cv <= 30:
    print("⚠️ Variabilidad moderada - Considerar estandarización para comparar")
else:
    print("⚠️ Alta variabilidad - REQUIERE estandarización")
    
    # OPCIÓN 1: Estandarización Z-Score (media=0, std=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['glucose_std'] = scaler.fit_transform(df[['glucose']])
    
    # OPCIÓN 2: Normalización Min-Max (rango 0-1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df['glucose_norm'] = scaler.fit_transform(df[['glucose']])
    
    # OPCIÓN 3: Robust Scaler (robusto a outliers)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df['glucose_robust'] = scaler.fit_transform(df[['glucose']])
    
    print("✅ Variable estandarizada")
```

##### **Según la Prueba de Normalidad**

```python
from scipy import stats

# Test de normalidad
stat, p_value = stats.shapiro(df['glucose'].sample(min(5000, len(df))))

print(f"Test de Shapiro-Wilk: p-valor = {p_value:.4f}")

# DECISIÓN:
if p_value > 0.05:
    print("✅ Distribución NORMAL")
    print("→ Usar: Media, Desviación Estándar, Correlación de Pearson, ANOVA")
else:
    print("⚠️ Distribución NO NORMAL")
    print("→ Usar: Mediana, IQR, Correlación de Spearman, Kruskal-Wallis")
    print("→ Considerar transformación o tests no paramétricos")
    
    # Aplicar transformación para normalizar
    df['glucose_log'] = np.log1p(df['glucose'])
    
    # Verificar si mejoró
    stat_new, p_new = stats.shapiro(df['glucose_log'].dropna().sample(min(5000, len(df))))
    print(f"\nDespués de transformación log: p-valor = {p_new:.4f}")
    if p_new > p_value:
        print("✅ Mejoró la normalidad")
```

##### **🎯 Ejemplo Completo: Preprocesamiento Automático**

```python
def preprocesar_variable_continua(df, columna):
    """
    Preprocesa automáticamente una variable continua según sus características
    """
    print(f"\n{'='*80}")
    print(f"🔧 PREPROCESAMIENTO: {columna}")
    print(f"{'='*80}")
    
    datos = df[columna].dropna()
    
    # 1. ASIMETRÍA
    skew = datos.skew()
    print(f"\n1️⃣ Asimetría: {skew:.2f}")
    
    if abs(skew) > 1:
        print("   → Asimetría alta - Aplicando transformación log")
        df[f'{columna}_transformado'] = np.log1p(df[columna])
        columna_usar = f'{columna}_transformado'
    else:
        print("   → Asimetría aceptable")
        columna_usar = columna
    
    # 2. OUTLIERS
    Q1 = datos.quantile(0.25)
    Q3 = datos.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((datos < Q1 - 1.5*IQR) | (datos > Q3 + 1.5*IQR)).sum()
    pct_outliers = outliers / len(datos) * 100
    
    print(f"\n2️⃣ Outliers: {outliers} ({pct_outliers:.2f}%)")
    
    if pct_outliers > 5:
        print("   → Muchos outliers - Aplicando winsorización")
        df = winsorizar(df, columna)
        columna_usar = f'{columna}_winsorizado'
    else:
        print("   → Outliers aceptables")
    
    # 3. ESTANDARIZACIÓN
    cv = (datos.std() / datos.mean()) * 100
    print(f"\n3️⃣ Coeficiente de Variación: {cv:.2f}%")
    
    if cv > 30:
        print("   → Alta variabilidad - Estandarizando")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[f'{columna}_std'] = scaler.fit_transform(df[[columna_usar]])
        print("   ✅ Variable lista para análisis multivariado")
    
    print(f"\n{'='*80}")
    print(f"✅ Preprocesamiento completado")
    print(f"Variables creadas: {[col for col in df.columns if columna in col]}")
    
    return df

# Aplicar a variables continuas de Framingham
for col in ['glucose', 'totChol', 'sysBP', 'BMI']:
    df = preprocesar_variable_continua(df, col)
```

---

### 2.2 Variables Cualitativas

#### 📊 Tabla de Frecuencias

```python
def analisis_univariado_cuali(df, columna):
    """Análisis completo de una variable categórica"""
    print(f"=== Análisis de {columna} ===\n")
    
    # Frecuencias absolutas y relativas
    freq_abs = df[columna].value_counts()
    freq_rel = df[columna].value_counts(normalize=True) * 100
    freq_acum = freq_rel.cumsum()
    
    tabla = pd.DataFrame({
        'Frecuencia': freq_abs,
        'Porcentaje': freq_rel.round(2),
        '% Acumulado': freq_acum.round(2)
    })
    
    print(tabla)
    print(f"\nCategorías distintas: {df[columna].nunique()}")
    print(f"Categoría más frecuente: {freq_abs.index[0]} ({freq_rel.iloc[0]:.2f}%)")
    
    return tabla
```

#### 📊 Visualizaciones

```python
def visualizar_univariado_cuali(df, columna):
    """Visualizaciones para variable categórica"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Gráfico de barras vertical
    df[columna].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Diagrama de Barras: {columna}')
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Gráfico de barras horizontal (útil para muchas categorías)
    df[columna].value_counts().plot(kind='barh', ax=axes[1], color='lightcoral')
    axes[1].set_title(f'Barras Horizontales: {columna}')
    axes[1].set_xlabel('Frecuencia')
    
    # 3. Gráfico de pastel
    df[columna].value_counts().plot(kind='pie', ax=axes[2], autopct='%1.1f%%')
    axes[2].set_title(f'Gráfico de Pastel: {columna}')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
```

---

#### 🔧 TRATAMIENTO DE VARIABLES CATEGÓRICAS

##### **Según Desbalance de Clases (Variables Binarias)**

```python
# Ejemplo con diabetes (binaria)
freq = df['diabetes'].value_counts()
prop_minoritaria = freq.min() / freq.sum()

print(f"Clase mayoritaria: {freq.idxmax()} ({freq.max()})")
print(f"Clase minoritaria: {freq.idxmin()} ({freq.min()})")
print(f"Proporción minoritaria: {prop_minoritaria:.2%}")
print(f"Ratio: 1:{freq.max()/freq.min():.1f}")

# DECISIÓN:
if prop_minoritaria < 0.1:
    print("\n⚠️ DESBALANCE CRÍTICO")
    print("Opciones de tratamiento:")
    
    # OPCIÓN 1: Oversampling (duplicar clase minoritaria)
    from sklearn.utils import resample
    
    # Separar clases
    df_mayoria = df[df['diabetes'] == 0]
    df_minoria = df[df['diabetes'] == 1]
    
    # Oversample de la minoría
    df_minoria_over = resample(df_minoria,
                               replace=True,
                               n_samples=len(df_mayoria),
                               random_state=42)
    
    # Combinar
    df_balanced = pd.concat([df_mayoria, df_minoria_over])
    print(f"✅ Oversampling: {len(df)} → {len(df_balanced)} filas")
    
    # OPCIÓN 2: Undersampling (reducir clase mayoritaria)
    df_mayoria_under = resample(df_mayoria,
                                replace=False,
                                n_samples=len(df_minoria),
                                random_state=42)
    
    df_balanced = pd.concat([df_mayoria_under, df_minoria])
    print(f"✅ Undersampling: {len(df)} → {len(df_balanced)} filas")
    
    # OPCIÓN 3: SMOTE (Synthetic Minority Over-sampling)
    from imblearn.over_sampling import SMOTE
    
    X = df[['age', 'BMI', 'glucose']]  # Features
    y = df['diabetes']  # Target
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"✅ SMOTE: {len(y)} → {len(y_balanced)} filas")
    
elif prop_minoritaria < 0.3:
    print("\n⚠️ Desbalance moderado - Considerar ajustar pesos en modelo")
else:
    print("\n✅ Clases balanceadas - No requiere tratamiento")
```

##### **Según Número de Categorías**

```python
# Ejemplo con education (4 categorías)
n_categorias = df['education'].nunique()
print(f"Número de categorías: {n_categorias}")

# DECISIÓN:
if n_categorias > 10:
    print("\n⚠️ Muchas categorías - Aplicar reducción")
    
    # OPCIÓN 1: Agrupar categorías de baja frecuencia
    freq = df['education'].value_counts(normalize=True)
    categorias_raras = freq[freq < 0.05].index
    
    df['education_agrupado'] = df['education'].apply(
        lambda x: 'Otros' if x in categorias_raras else x
    )
    print(f"✅ Categorías reducidas: {n_categorias} → {df['education_agrupado'].nunique()}")
    
    # OPCIÓN 2: Mantener solo Top N categorías
    top_n = 5
    top_categorias = df['education'].value_counts().nlargest(top_n).index
    df['education_top'] = df['education'].apply(
        lambda x: x if x in top_categorias else 'Otros'
    )
    print(f"✅ Mantener solo Top {top_n} categorías")

elif n_categorias == 2:
    print("\n✅ Variable binaria")
    
    # Verificar codificación (0/1 o texto)
    valores = df['education'].unique()
    if not all(v in [0, 1] for v in valores):
        print("⚠️ No está codificada numéricamente - Codificar:")
        
        # OPCIÓN 1: Label Encoding (0, 1)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['education_encoded'] = le.fit_transform(df['education'])
        print(f"✅ Codificación: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # OPCIÓN 2: Manual
        df['education_encoded'] = df['education'].map({'No': 0, 'Sí': 1})

else:
    print(f"\n✅ {n_categorias} categorías - Cantidad manejable")
```

##### **Codificación de Variables Categóricas (Para Modelado)**

```python
# OPCIÓN 1: One-Hot Encoding (para variables nominales)
# ✅ Usar cuando: No hay orden entre categorías
# Ejemplo: género, país, tipo_sangre

# Método simple con pandas
df_encoded = pd.get_dummies(df, columns=['education'], prefix='edu')
print(f"Columnas originales: {df.shape[1]}")
print(f"Columnas después de encoding: {df_encoded.shape[1]}")

# Método con sklearn (más control)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' evita multicolinealidad
encoded = encoder.fit_transform(df[['education']])
encoded_df = pd.DataFrame(
    encoded, 
    columns=encoder.get_feature_names_out(['education'])
)
df_encoded = pd.concat([df, encoded_df], axis=1)
print("✅ One-Hot Encoding aplicado")


# OPCIÓN 2: Label Encoding (para variables ordinales)
# ✅ Usar cuando: HAY orden entre categorías
# Ejemplo: education (1=primaria, 2=secundaria, 3=universidad, 4=posgrado)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
print(f"✅ Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# OPCIÓN 3: Ordinal Encoding (orden personalizado)
# ✅ Mejor para variables con orden específico

from sklearn.preprocessing import OrdinalEncoder

# Definir el orden
orden = [['Primaria', 'Secundaria', 'Universidad', 'Posgrado']]

encoder = OrdinalEncoder(categories=orden)
df['education_ordinal'] = encoder.fit_transform(df[['education']])
print("✅ Ordinal Encoding con orden personalizado")


# OPCIÓN 4: Frequency Encoding (por frecuencia)
# ✅ Útil cuando hay muchas categorías

freq = df['education'].value_counts(normalize=True).to_dict()
df['education_freq'] = df['education'].map(freq)
print("✅ Frequency Encoding aplicado")


# OPCIÓN 5: Target Encoding (por relación con variable objetivo)
# ⚠️ Usar con cuidado (puede causar overfitting)

# Codificar según la media del target
target_means = df.groupby('education')['TenYearCHD'].mean().to_dict()
df['education_target'] = df['education'].map(target_means)
print("✅ Target Encoding aplicado")
```

##### **🎯 Ejemplo Completo: Preprocesar Variables Categóricas de Framingham**

```python
def preprocesar_categoricas_framingham(df):
    """
    Preprocesa variables categóricas del dataset Framingham
    """
    print("=" * 80)
    print(" PREPROCESAMIENTO DE VARIABLES CATEGÓRICAS ".center(80, "="))
    print("=" * 80)
    
    # 1. BINARIAS: Ya están codificadas como 0/1 ✅
    binarias = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 
                'prevalentHyp', 'diabetes', 'TenYearCHD']
    
    print("\n1️⃣ VARIABLES BINARIAS")
    for col in binarias:
        if col in df.columns:
            freq = df[col].value_counts()
            print(f"   {col}: {freq.to_dict()}")
            
            # Verificar desbalance
            prop = freq.min() / freq.sum()
            if prop < 0.1:
                print(f"      ⚠️ Desbalance: {prop:.2%} - Considerar balanceo")
    
    # 2. EDUCATION (ordinal): Codificar con orden
    print("\n2️⃣ EDUCATION (Variable Ordinal)")
    if 'education' in df.columns:
        # Ya es numérica 1-4, verificar y rellenar nulos
        print(f"   Valores únicos: {sorted(df['education'].dropna().unique())}")
        
        # Imputar nulos con la moda
        if df['education'].isnull().sum() > 0:
            moda = df['education'].mode()[0]
            df['education'].fillna(moda, inplace=True)
            print(f"   ✅ {df['education'].isnull().sum()} nulos imputados con moda = {moda}")
        
        # Crear versión estandarizada para algunos análisis
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df['education_std'] = scaler.fit_transform(df[['education']])
        print("   ✅ Versión estandarizada creada")
    
    # 3. ONE-HOT ENCODING (para ciertos análisis)
    print("\n3️⃣ ONE-HOT ENCODING (Opcional)")
    df_dummies = pd.get_dummies(df, columns=['education'], prefix='edu', drop_first=True)
    print(f"   ✅ Dataset con dummies: {df_dummies.shape[1]} columnas")
    
    print("\n" + "=" * 80)
    print("✅ Preprocesamiento completado")
    
    return df

# Aplicar
df = preprocesar_categoricas_framingham(df)
```

---

## 3. Análisis Bivariado y Correlatividad

Objetivo: **Estudiar la relación simultánea entre dos variables** $(X, Y)$.

### 3.1 Cuantitativa vs Cuantitativa

#### 🔗 Coeficiente de Correlación de Pearson

Mide la **relación lineal** entre dos variables:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

**Propiedades:**
- Rango: $-1 \leq r \leq 1$
- $r = 1$: Correlación positiva perfecta
- $r = -1$: Correlación negativa perfecta
- $r = 0$: No hay correlación lineal
- $|r| < 0.3$: Correlación débil
- $0.3 \leq |r| < 0.7$: Correlación moderada
- $|r| \geq 0.7$: Correlación fuerte

**⚠️ Importante:** La correlación NO implica causalidad

```python
def analisis_correlacion(df, var1, var2):
    """Análisis de correlación entre dos variables cuantitativas"""
    # Correlación de Pearson
    r_pearson, p_pearson = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
    
    # Correlación de Spearman (no asume linealidad)
    r_spearman, p_spearman = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
    
    print(f"=== Correlación entre {var1} y {var2} ===\n")
    print(f"Pearson: r = {r_pearson:.3f} (p-valor = {p_pearson:.4f})")
    print(f"  → {'Significativa' if p_pearson < 0.05 else 'No significativa'} al 5%")
    print(f"Spearman: ρ = {r_spearman:.3f} (p-valor = {p_spearman:.4f})")
    
    # Interpretación
    if abs(r_pearson) < 0.3:
        fuerza = "débil"
    elif abs(r_pearson) < 0.7:
        fuerza = "moderada"
    else:
        fuerza = "fuerte"
    
    direccion = "positiva" if r_pearson > 0 else "negativa"
    print(f"\n📊 Correlación {fuerza} {direccion}")
```

#### 📊 Visualización

```python
def visualizar_bivariado_cuanti(df, var1, var2):
    """Scatter plot con línea de regresión"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot básico
    axes[0].scatter(df[var1], df[var2], alpha=0.5)
    axes[0].set_xlabel(var1)
    axes[0].set_ylabel(var2)
    axes[0].set_title(f'Dispersión: {var1} vs {var2}')
    
    # Scatter plot con regresión
    sns.regplot(x=var1, y=var2, data=df, ax=axes[1], 
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[1].set_title(f'Dispersión + Regresión Lineal')
    
    plt.tight_layout()
    plt.show()

# Matriz de correlación (múltiples variables)
def matriz_correlacion(df, columnas_numericas):
    """Heatmap de correlaciones"""
    corr_matrix = df[columnas_numericas].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=1)
    plt.title('Matriz de Correlación de Pearson', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix
```

---

### 3.2 Cualitativa vs Cualitativa

#### 🎲 Prueba de Chi-Cuadrado ($\chi^2$)

Evalúa si existe **asociación significativa** entre dos variables categóricas.

**Hipótesis:**
- $H_0$: Las variables son independientes (no hay relación)
- $H_1$: Las variables están asociadas

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Donde:
- $O_{ij}$: Frecuencia observada
- $E_{ij}$: Frecuencia esperada bajo independencia
- $r$: Número de filas
- $c$: Número de columnas

**Decisión:** Si $p\text{-valor} < 0.05$ → Rechazamos $H_0$ (hay asociación)

```python
def analisis_chi_cuadrado(df, var1, var2):
    """Prueba de independencia Chi-Cuadrado"""
    # Tabla de contingencia
    tabla = pd.crosstab(df[var1], df[var2], margins=True)
    print(f"=== Tabla de Contingencia: {var1} vs {var2} ===\n")
    print(tabla)
    
    # Chi-Cuadrado
    chi2, p_valor, gl, frecuencias_esperadas = stats.chi2_contingency(
        pd.crosstab(df[var1], df[var2])
    )
    
    print(f"\n=== Prueba Chi-Cuadrado ===")
    print(f"Estadístico χ²: {chi2:.3f}")
    print(f"Grados de libertad: {gl}")
    print(f"P-valor: {p_valor:.4f}")
    print(f"Conclusión: {'Hay asociación significativa' if p_valor < 0.05 else 'No hay asociación significativa'} (α=0.05)")
    
    # V de Cramér (fuerza de la asociación)
    n = df[[var1, var2]].dropna().shape[0]
    min_dim = min(tabla.shape[0]-1, tabla.shape[1]-1)
    v_cramer = np.sqrt(chi2 / (n * min_dim))
    print(f"V de Cramér: {v_cramer:.3f} (0=sin asociación, 1=asociación perfecta)")
```

#### 📊 Visualizaciones

```python
def visualizar_bivariado_cuali(df, var1, var2):
    """Visualizaciones para variables categóricas"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Gráfico de barras agrupadas
    tabla = pd.crosstab(df[var1], df[var2])
    tabla.plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Barras Agrupadas: {var1} vs {var2}')
    axes[0].legend(title=var2)
    axes[0].set_ylabel('Frecuencia')
    
    # 2. Gráfico de barras apiladas (100%)
    tabla_pct = pd.crosstab(df[var1], df[var2], normalize='index') * 100
    tabla_pct.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title(f'Barras Apiladas (100%): {var1} vs {var2}')
    axes[1].set_ylabel('Porcentaje')
    axes[1].legend(title=var2)
    
    # 3. Heatmap de frecuencias
    sns.heatmap(tabla, annot=True, fmt='d', cmap='YlGnBu', ax=axes[2])
    axes[2].set_title(f'Heatmap de Frecuencias: {var1} vs {var2}')
    
    plt.tight_layout()
    plt.show()
```

---

### 3.3 Cualitativa vs Cuantitativa

#### 📊 Análisis de Variabilidad entre Grupos

Comparar la distribución de una variable cuantitativa entre diferentes categorías.

**Técnicas estadísticas:**
- **2 grupos:** Prueba t de Student
- **≥ 3 grupos:** ANOVA (Análisis de Varianza)

```python
def analisis_cuali_cuanti(df, var_categorica, var_numerica):
    """Análisis de variable numérica por grupos categóricos"""
    print(f"=== Análisis de {var_numerica} por {var_categorica} ===\n")
    
    # Estadísticas descriptivas por grupo
    resumen = df.groupby(var_categorica)[var_numerica].describe()
    print("Estadísticas por grupo:")
    print(resumen)
    
    # Prueba de diferencias
    grupos = [group[var_numerica].dropna() for name, group in df.groupby(var_categorica)]
    
    if len(grupos) == 2:
        # Test t de Student
        t_stat, p_valor = stats.ttest_ind(grupos[0], grupos[1])
        print(f"\nPrueba t de Student:")
        print(f"Estadístico t: {t_stat:.3f}")
        print(f"P-valor: {p_valor:.4f}")
        print(f"Conclusión: {'Hay diferencia significativa' if p_valor < 0.05 else 'No hay diferencia significativa'}")
    
    elif len(grupos) >= 3:
        # ANOVA
        f_stat, p_valor = stats.f_oneway(*grupos)
        print(f"\nANOVA (Análisis de Varianza):")
        print(f"Estadístico F: {f_stat:.3f}")
        print(f"P-valor: {p_valor:.4f}")
        print(f"Conclusión: {'Al menos un grupo es diferente' if p_valor < 0.05 else 'No hay diferencias significativas'}")
```

#### 📊 Visualizaciones

```python
def visualizar_cuali_cuanti(df, var_categorica, var_numerica):
    """Visualizaciones para Cualitativa vs Cuantitativa"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Boxplots comparativos
    sns.boxplot(x=var_categorica, y=var_numerica, data=df, ax=axes[0])
    axes[0].set_title(f'Boxplots: {var_numerica} por {var_categorica}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Violin plots (distribución completa)
    sns.violinplot(x=var_categorica, y=var_numerica, data=df, ax=axes[1])
    axes[1].set_title(f'Violin Plots: {var_numerica} por {var_categorica}')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Histogramas superpuestos
    for categoria in df[var_categorica].unique():
        subset = df[df[var_categorica] == categoria][var_numerica]
        axes[2].hist(subset, alpha=0.5, label=categoria, bins=20)
    axes[2].set_title(f'Histogramas: {var_numerica} por {var_categorica}')
    axes[2].set_xlabel(var_numerica)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
```

---

## 4. Análisis Multivariado

### 4.1 Análisis de Componentes Principales (ACP/PCA)

**Objetivo:** Reducir la dimensionalidad transformando variables correlacionadas en un nuevo conjunto de variables **independientes** (componentes principales) que capturan la mayor varianza posible.

#### 🎯 Cuándo usar ACP:

✅ **Usar cuando:**
- Tienes muchas variables cuantitativas correlacionadas
- Necesitas visualizar datos de alta dimensión
- Quieres eliminar multicolinealidad
- Buscas identificar patrones o estructuras latentes

❌ **No usar cuando:**
- Variables categóricas (usar MCA - Multiple Correspondence Analysis)
- Pocas variables (< 3)
- Variables no correlacionadas
- Necesitas interpretabilidad directa

#### 📐 Fundamentos Matemáticos

**Componentes principales:**
- Combinaciones lineales de las variables originales
- Ortogonales entre sí (no correlacionadas)
- Ordenadas por varianza explicada

$$PC_1 = w_{11}X_1 + w_{12}X_2 + ... + w_{1p}X_p$$
$$PC_2 = w_{21}X_1 + w_{22}X_2 + ... + w_{2p}X_p$$

**Criterios de selección de componentes:**
1. **Criterio de Kaiser:** Retener componentes con eigenvalor > 1
2. **Varianza explicada:** Retener hasta alcanzar 70-90% de varianza acumulada
3. **Scree Plot:** Buscar el "codo" en el gráfico

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def analisis_pca(df, columnas_numericas, n_componentes=None):
    """
    Análisis de Componentes Principales completo
    """
    # 1. PREPARACIÓN: Estandarizar datos (CRUCIAL para PCA)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df[columnas_numericas].dropna())
    
    # 2. APLICAR PCA
    if n_componentes is None:
        pca = PCA()  # Todos los componentes
    else:
        pca = PCA(n_components=n_componentes)
    
    componentes = pca.fit_transform(datos_escalados)
    
    # 3. ANÁLISIS DE RESULTADOS
    print("=== ANÁLISIS DE COMPONENTES PRINCIPALES ===\n")
    print(f"Número de componentes: {pca.n_components_}")
    print(f"\nVarianza explicada por componente:")
    
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {var*100:.2f}%")
    
    print(f"\nVarianza acumulada:")
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    for i, var in enumerate(var_acum, 1):
        print(f"  PC1 a PC{i}: {var*100:.2f}%")
    
    # 4. MATRIZ DE CARGAS (loadings)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i}' for i in range(1, pca.n_components_+1)],
        index=columnas_numericas
    )
    print("\n=== MATRIZ DE CARGAS (Loadings) ===")
    print(loadings.round(3))
    
    # 5. INTERPRETACIÓN DE COMPONENTES
    print("\n=== INTERPRETACIÓN DE COMPONENTES ===")
    for i in range(min(3, pca.n_components_)):
        print(f"\nPC{i+1} (explica {pca.explained_variance_ratio_[i]*100:.2f}%):")
        cargas = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print("  Variables más influyentes:")
        for var, carga in cargas.head(3).items():
            signo = '+' if loadings.loc[var, f'PC{i+1}'] > 0 else '-'
            print(f"    {signo} {var}: {carga:.3f}")
    
    return pca, componentes, loadings
```

#### 📊 Visualizaciones PCA

```python
def visualizar_pca(pca, componentes, loadings, df=None, variable_color=None):
    """Visualizaciones completas del PCA"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. SCREE PLOT (varianza explicada)
    ax1 = plt.subplot(2, 3, 1)
    varianza = pca.explained_variance_ratio_ * 100
    ax1.bar(range(1, len(varianza)+1), varianza, alpha=0.7, color='steelblue')
    ax1.plot(range(1, len(varianza)+1), varianza, 'ro-')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada (%)')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    
    # 2. VARIANZA ACUMULADA
    ax2 = plt.subplot(2, 3, 2)
    var_acum = np.cumsum(varianza)
    ax2.plot(range(1, len(var_acum)+1), var_acum, 'go-', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', label='70%')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada (%)')
    ax2.set_title('Varianza Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. EIGENVALORES
    ax3 = plt.subplot(2, 3, 3)
    eigenvalores = pca.explained_variance_
    ax3.bar(range(1, len(eigenvalores)+1), eigenvalores, color='coral')
    ax3.axhline(y=1, color='r', linestyle='--', label='Criterio Kaiser (=1)')
    ax3.set_xlabel('Componente Principal')
    ax3.set_ylabel('Eigenvalor')
    ax3.set_title('Eigenvalores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. BIPLOT (PC1 vs PC2)
    ax4 = plt.subplot(2, 3, 4)
    if df is not None and variable_color is not None:
        scatter = ax4.scatter(componentes[:, 0], componentes[:, 1], 
                             c=df[variable_color], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label=variable_color)
    else:
        ax4.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('Biplot: PC1 vs PC2')
    ax4.grid(True, alpha=0.3)
    
    # 5. CÍRCULO DE CORRELACIONES (loadings)
    ax5 = plt.subplot(2, 3, 5)
    circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax5.add_patch(circulo)
    
    for i, var in enumerate(loadings.index):
        ax5.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
                 head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax5.text(loadings.iloc[i, 0]*1.1, loadings.iloc[i, 1]*1.1, var,
                fontsize=9, ha='center')
    
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax5.set_title('Círculo de Correlaciones')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. HEATMAP DE LOADINGS
    ax6 = plt.subplot(2, 3, 6)
    sns.heatmap(loadings.iloc[:, :min(5, loadings.shape[1])], 
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                ax=ax6, cbar_kws={'label': 'Carga'})
    ax6.set_title('Heatmap de Cargas (Loadings)')
    
    plt.tight_layout()
    plt.show()
```

#### 🔍 Ejemplo Completo de Uso

```python
# Supongamos un dataset con variables de salud
columnas_num = ['edad', 'presion_sistolica', 'presion_diastolica', 
                'colesterol', 'glucosa', 'imc']

# Ejecutar PCA
pca, componentes, loadings = analisis_pca(df, columnas_num)

# Decidir cuántos componentes retener (ej: 70% varianza)
var_acum = np.cumsum(pca.explained_variance_ratio_)
n_componentes_70 = np.argmax(var_acum >= 0.70) + 1
print(f"\nPara explicar 70% de varianza: {n_componentes_70} componentes")

# Reajustar con número óptimo
pca_final, componentes_final, loadings_final = analisis_pca(
    df, columnas_num, n_componentes=n_componentes_70
)

# Visualizar
visualizar_pca(pca_final, componentes_final, loadings_final, 
               df=df, variable_color='enfermedad_cardiaca')

# Crear dataframe con componentes
df_pca = pd.DataFrame(
    componentes_final,
    columns=[f'PC{i}' for i in range(1, n_componentes_70+1)]
)
df_pca = pd.concat([df_pca, df.reset_index(drop=True)], axis=1)
```

---

### 4.2 Matriz de Covarianza y Correlación

```python
def analisis_covarianza(df, columnas):
    """Análisis de matriz de covarianza y correlación"""
    
    # Matriz de Covarianza
    cov_matrix = df[columnas].cov()
    print("=== MATRIZ DE COVARIANZA ===")
    print(cov_matrix.round(2))
    
    # Matriz de Correlación
    corr_matrix = df[columnas].corr()
    print("\n=== MATRIZ DE CORRELACIÓN ===")
    print(corr_matrix.round(3))
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Matriz de Covarianza')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Matriz de Correlación')
    
    plt.tight_layout()
    plt.show()
    
    return cov_matrix, corr_matrix
```

---

## 5. Implementación en Python

### 5.1 Script Completo de EDN

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExploracionDatos:
    """
    Clase para Exploración de Datos de Negocio (EDN)
    """
    
    def __init__(self, df):
        self.df = df
        self.col_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        self.col_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def resumen_general(self):
        """Resumen general del dataset"""
        print("="*60)
        print(" RESUMEN GENERAL DEL DATASET ".center(60, '='))
        print("="*60)
        print(f"\n📊 Dimensiones: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        print(f"📈 Variables numéricas: {len(self.col_numericas)}")
        print(f"📝 Variables categóricas: {len(self.col_categoricas)}")
        print(f"\n💾 Memoria: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== VALORES NULOS ===")
        nulos = self.df.isnull().sum()
        nulos_pct = (nulos / len(self.df) * 100).round(2)
        tabla_nulos = pd.DataFrame({
            'Nulos': nulos[nulos > 0],
            'Porcentaje': nulos_pct[nulos > 0]
        }).sort_values('Nulos', ascending=False)
        
        if len(tabla_nulos) > 0:
            print(tabla_nulos)
        else:
            print("✅ No hay valores nulos")
        
        print("\n=== DUPLICADOS ===")
        duplicados = self.df.duplicated().sum()
        print(f"Filas duplicadas: {duplicados} ({duplicados/len(self.df)*100:.2f}%)")
        
    def limpiar_categoricas(self, columna, reemplazos=None, umbral_raros=0.05):
        """Limpieza de variables categóricas"""
        print(f"\n=== Limpiando {columna} ===")
        
        # Normalizar
        self.df[columna] = self.df[columna].str.strip().str.lower()
        
        # Reemplazar nulos enmascarados
        self.df[columna].replace(['?', 'n/a', 'null', 'unknown', '-', ' '], 
                                 np.nan, inplace=True)
        
        # Aplicar diccionario de reemplazos
        if reemplazos:
            self.df[columna].replace(reemplazos, inplace=True)
        
        # Agrupar categorías raras
        freq = self.df[columna].value_counts(normalize=True)
        raras = freq[freq < umbral_raros].index
        if len(raras) > 0:
            print(f"Agrupando {len(raras)} categorías raras en 'Otros'")
            self.df[columna] = self.df[columna].apply(
                lambda x: 'otros' if x in raras else x
            )
        
        print(f"Categorías finales: {self.df[columna].nunique()}")
        
    def detectar_outliers(self, columna, metodo='tukey', k=1.5):
        """Detecta outliers en variable numérica"""
        if metodo == 'tukey':
            Q1 = self.df[columna].quantile(0.25)
            Q3 = self.df[columna].quantile(0.75)
            IQR = Q3 - Q1
            li = Q1 - k * IQR
            ls = Q3 + k * IQR
            outliers = self.df[(self.df[columna] < li) | (self.df[columna] > ls)]
        
        elif metodo == 'zscore':
            z = np.abs(stats.zscore(self.df[columna].dropna()))
            outliers = self.df[z > 3]
        
        print(f"\n=== Outliers en {columna} ({metodo}) ===")
        print(f"Detectados: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
        
        return outliers
    
    def analisis_univariado(self, tipo='todas'):
        """Análisis univariado completo"""
        print("\n" + "="*60)
        print(" ANÁLISIS UNIVARIADO ".center(60, '='))
        print("="*60)
        
        if tipo in ['todas', 'numericas']:
            print("\n📊 VARIABLES NUMÉRICAS\n")
            print(self.df[self.col_numericas].describe().round(2))
            
        if tipo in ['todas', 'categoricas']:
            print("\n📝 VARIABLES CATEGÓRICAS\n")
            for col in self.col_categoricas:
                print(f"\n--- {col} ---")
                print(self.df[col].value_counts().head())
    
    def matriz_correlacion(self, umbral=0.7):
        """Matriz de correlación con alertas de multicolinealidad"""
        corr = self.df[self.col_numericas].corr()
        
        # Identificar correlaciones altas
        alta_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > umbral:
                    alta_corr.append((
                        corr.columns[i],
                        corr.columns[j],
                        corr.iloc[i, j]
                    ))
        
        if alta_corr:
            print(f"\n⚠️  ALERTAS: Correlaciones > {umbral}")
            for var1, var2, r in alta_corr:
                print(f"  • {var1} ↔ {var2}: r = {r:.3f}")
        
        # Visualizar
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True, linewidths=1)
        plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return corr
    
    def ejecutar_pca(self, n_componentes=None):
        """Análisis de Componentes Principales"""
        scaler = StandardScaler()
        datos_std = scaler.fit_transform(self.df[self.col_numericas].dropna())
        
        pca = PCA(n_components=n_componentes)
        componentes = pca.fit_transform(datos_std)
        
        print("\n" + "="*60)
        print(" ANÁLISIS DE COMPONENTES PRINCIPALES ".center(60, '='))
        print("="*60)
        
        print(f"\nVarianza explicada por componente:")
        for i, var in enumerate(pca.explained_variance_ratio_, 1):
            print(f"  PC{i}: {var*100:.2f}%")
        
        var_acum = np.cumsum(pca.explained_variance_ratio_) * 100
        print(f"\nVarianza acumulada: {var_acum[-1]:.2f}%")
        
        return pca, componentes

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv('framingham.csv')
    
    # Crear objeto de exploración
    edn = ExploracionDatos(df)
    
    # 1. Resumen general
    edn.resumen_general()
    
    # 2. Limpieza de categóricas
    if 'genero' in df.columns:
        edn.limpiar_categoricas('genero', 
                               reemplazos={'m': 'masculino', 'f': 'femenino'})
    
    # 3. Detección de outliers
    if 'edad' in df.columns:
        outliers_edad = edn.detectar_outliers('edad', metodo='tukey')
    
    # 4. Análisis univariado
    edn.analisis_univariado(tipo='todas')
    
    # 5. Matriz de correlación
    corr_matrix = edn.matriz_correlacion(umbral=0.7)
    
    # 6. PCA
    pca, componentes = edn.ejecutar_pca(n_componentes=None)
```

---

### 5.2 Checklist de Exploración

```markdown
## ✅ CHECKLIST DE EXPLORACIÓN DE DATOS

### Fase 1: Calidad y Limpieza
- [ ] Verificar dimensiones del dataset
- [ ] Identificar tipos de variables (numéricas/categóricas)
- [ ] Detectar valores nulos
- [ ] Buscar duplicados
- [ ] Normalizar variables categóricas (mayúsculas, espacios)
- [ ] Unificar categorías inconsistentes
- [ ] Detectar outliers en variables numéricas
- [ ] Validar rangos lógicos

### Fase 2: Análisis Univariado
- [ ] Calcular estadísticas descriptivas (numéricas)
- [ ] Crear histogramas y boxplots
- [ ] Tablas de frecuencia (categóricas)
- [ ] Gráficos de barras
- [ ] Evaluar normalidad (QQ-plots)
- [ ] Identificar asimetrías y curtosis

### Fase 3: Análisis Bivariado
- [ ] Matriz de correlación (Pearson)
- [ ] Scatter plots (cuanti vs cuanti)
- [ ] Pruebas Chi-Cuadrado (cuali vs cuali)
- [ ] Tablas de contingencia
- [ ] ANOVA / T-test (cuali vs cuanti)
- [ ] Boxplots comparativos

### Fase 4: Análisis Multivariado
- [ ] Estandarizar variables
- [ ] Aplicar PCA
- [ ] Determinar número óptimo de componentes
- [ ] Interpretar loadings
- [ ] Visualizar biplots
- [ ] Analizar matriz de covarianza
```

---

## 📚 Referencias y Recursos

### Librerías Python
- **Pandas:** Manipulación de datos
- **NumPy:** Computación numérica
- **Matplotlib/Seaborn:** Visualización
- **Scipy:** Estadística
- **Scikit-learn:** Machine Learning y PCA

### Documentación
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)

### Conceptos Clave
- **EDA (Exploratory Data Analysis):** Proceso iterativo de análisis visual y estadístico
- **Data Quality:** Integridad, consistencia, completitud, validez
- **Feature Engineering:** Creación y transformación de variables
- **Dimensionality Reduction:** Técnicas para reducir número de variables

---

## 🎯 Conclusiones

La Exploración de Datos de Negocio (EDN) es un proceso **iterativo y crítico** que permite:

1. ✅ **Garantizar calidad:** Identificar y corregir inconsistencias
2. 📊 **Comprender distribuciones:** Conocer el comportamiento de cada variable
3. 🔗 **Descubrir relaciones:** Identificar correlaciones y dependencias
4. 🎯 **Reducir complejidad:** Simplificar modelos mediante PCA
5. 💡 **Generar insights:** Extraer conocimiento accionable del negocio

> **Recuerda:** Un buen análisis exploratorio es la base de cualquier proyecto de ciencia de datos exitoso.

---

*Documento creado para el Trabajo Práctico 1 - Digital House Data Science*
*Última actualización: 11 de enero de 2026*
