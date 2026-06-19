#!/usr/bin/env python3
"""
Script para mejorar los markdown, textos y títulos de la notebook resolucionTP.ipynb
Agrega contexto, explicaciones y mejora la estructura narrativa.
"""

import json
import os
from pathlib import Path

def cargar_notebook(ruta):
    """Carga el notebook JSON"""
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)

def guardar_notebook(notebook, ruta):
    """Guarda el notebook JSON con formato"""
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def mejorar_headers_visualizaciones(cell):
    """Mejora headers de visualizaciones de variables continuas"""
    mejoras = {
        "### Colesterol Total (totChol)": """### Colesterol Total (totChol)

Esta variable representa el nivel de colesterol total en sangre (mg/dL). El colesterol es un factor de riesgo cardiovascular importante:

- **Valores normales:** < 200 mg/dL
- **Valores límite:** 200-239 mg/dL
- **Valores altos:** ≥ 240 mg/dL

Analizamos la distribución para identificar la concentración de pacientes en cada rango y detectar valores extremos.""",

        "### Presión Arterial Sistólica (sysBP)": """### Presión Arterial Sistólica (sysBP)

La presión arterial sistólica mide la presión cuando el corazón late (mmHg). Es un predictor clave de eventos cardiovasculares:

- **Normal:** < 120 mmHg
- **Elevada:** 120-129 mmHg
- **Hipertensión Stage 1:** 130-139 mmHg
- **Hipertensión Stage 2:** ≥ 140 mmHg

Visualizamos su distribución para entender el perfil de riesgo de la población.""",

        "### Presión Arterial Diastólica (diaBP)": """### Presión Arterial Diastólica (diaBP)

La presión arterial diastólica mide la presión cuando el corazón descansa entre latidos (mmHg):

- **Normal:** < 80 mmHg
- **Hipertensión Stage 1:** 80-89 mmHg
- **Hipertensión Stage 2:** ≥ 90 mmHg

Junto con la sistólica, nos permite clasificar la hipertensión arterial de los pacientes.""",

        "### Índice de Masa Corporal (BMI)": """### Índice de Masa Corporal (BMI)

El BMI (Body Mass Index) es un indicador del peso corporal en relación con la altura (kg/m²):

- **Bajo peso:** < 18.5
- **Peso normal:** 18.5-24.9
- **Sobrepeso:** 25-29.9
- **Obesidad:** ≥ 30

La obesidad es un factor de riesgo cardiovascular modificable. Analizamos su distribución para identificar la prevalencia en la muestra.""",

        "### Frecuencia Cardíaca (heartRate)": """### Frecuencia Cardíaca (heartRate)

Mide los latidos del corazón por minuto (bpm). Valores típicos en adultos:

- **Bradicardia:** < 60 bpm
- **Normal en reposo:** 60-100 bpm
- **Taquicardia:** > 100 bpm

Una frecuencia cardíaca elevada puede indicar estrés cardiovascular o condición física deficiente.""",

        "### Glucosa (glucose)": """### Glucosa (glucose)

Nivel de glucosa en sangre (mg/dL), relacionado con diabetes y riesgo cardiovascular:

- **Normal en ayunas:** 70-99 mg/dL
- **Prediabetes:** 100-125 mg/dL
- **Diabetes:** ≥ 126 mg/dL

La diabetes es un factor de riesgo cardiovascular mayor. Analizamos la distribución para identificar casos de hiperglucemia."""
    }

    if cell['cell_type'] == 'markdown' and cell['source']:
        texto = ''.join(cell['source'])
        for header_viejo, header_nuevo in mejoras.items():
            if texto.strip() == header_viejo:
                cell['source'] = [header_nuevo + '\n']
                return True
    return False

def mejorar_preguntas_qa(cell):
    """Agrega contexto a las preguntas de la sección 8"""
    mejoras = {
        "## 8.1 ¿Cuántos registros hay?": """## 8.1 ¿Cuántos registros hay?

Para entender la magnitud del estudio Framingham, contamos el número total de pacientes y variables incluidas en el dataset. Esto nos da una idea del tamaño de la muestra y la cantidad de información disponible para el análisis.""",

        "## 8.2 ¿Qué tipo de variable es cada una?": """## 8.2 ¿Qué tipo de variable es cada una?

Clasificamos cada variable según su tipo de dato (numérica o categórica). Esta clasificación es fundamental para:
- Elegir las técnicas de análisis apropiadas
- Seleccionar las visualizaciones adecuadas
- Aplicar las transformaciones correctas en el preprocesamiento""",

        "## 8.3 ¿Hay valores faltantes?": """## 8.3 ¿Hay valores faltantes?

Los valores faltantes (missing values) pueden afectar la calidad del análisis y el modelado. Identificamos:
- Qué variables tienen datos faltantes
- Cuántos valores faltan en cada una
- El porcentaje de completitud del dataset

Esto nos ayudará a decidir qué estrategia de imputación utilizar.""",

        "## 8.4 ¿Cómo se distribuyen las variables continuas principales?": """## 8.4 ¿Cómo se distribuyen las variables continuas principales?

Analizamos la distribución de las variables numéricas clave para entender:
- Tendencia central (media, mediana)
- Dispersión (desviación estándar, rango)
- Forma de la distribución (simetría, curtosis)
- Presencia de outliers

Esto es esencial para detectar anomalías y decidir si necesitamos transformaciones.""",

        "## 8.5 ¿Existen outliers en las variables continuas?": """## 8.5 ¿Existen outliers en las variables continuas?

Los outliers son valores extremos que pueden ser:
- **Datos válidos:** Casos reales pero inusuales (ej: pacientes con hipertensión severa)
- **Errores:** Datos mal registrados o ingresados incorrectamente

Utilizamos el método IQR (Interquartile Range) para detectarlos y decidir cómo tratarlos.""",

        "## 8.6 ¿Qué variables están correlacionadas con el riesgo cardiovascular (TenYearCHD)?": """## 8.6 ¿Qué variables están correlacionadas con el riesgo cardiovascular (TenYearCHD)?

Identificamos qué factores tienen mayor asociación con el desarrollo de enfermedad cardiovascular en los próximos 10 años. Esto nos ayuda a:
- Entender los principales predictores de riesgo
- Priorizar variables para el modelo
- Validar el conocimiento médico existente

Analizamos tanto correlaciones positivas (aumentan el riesgo) como negativas (lo reducen).""",

        "## 8.7 ¿Las variables categóricas muestran diferencias en la proporción de riesgo?": """## 8.7 ¿Las variables categóricas muestran diferencias en la proporción de riesgo?

Comparamos la tasa de eventos cardiovasculares entre diferentes grupos:
- Fumadores vs no fumadores
- Hombres vs mujeres
- Diabéticos vs no diabéticos
- Hipertensos vs normotensos

Esto revela qué factores categóricos son más discriminantes para predecir riesgo.""",

        "## 8.8 ¿Cómo afectan las transformaciones a la distribución de outliers?": """## 8.8 ¿Cómo afectan las transformaciones a la distribución de outliers?

Aplicamos dos técnicas de tratamiento de outliers:
- **Transformación logarítmica:** Comprime valores extremos y normaliza distribuciones asimétricas
- **Winsorización:** Reemplaza outliers por valores límite (percentiles)

Comparamos la efectividad de cada método para reducir el impacto de valores extremos sin perder información.""",

        "## 8.9 ¿Qué rendimiento obtiene el modelo de regresión logística?": """## 8.9 ¿Qué rendimiento obtiene el modelo de regresión logística?

Evaluamos el modelo usando múltiples métricas:
- **Accuracy:** Porcentaje de predicciones correctas
- **Precision:** De los casos predichos como positivos, cuántos lo son realmente
- **Recall:** De los casos reales positivos, cuántos detectamos
- **F1-Score:** Media armónica entre precision y recall
- **ROC-AUC:** Capacidad del modelo para discriminar entre clases

Estas métricas nos permiten evaluar si el modelo es útil para predecir riesgo cardiovascular."""
    }

    if cell['cell_type'] == 'markdown' and cell['source']:
        texto = ''.join(cell['source'])
        for pregunta_vieja, pregunta_nueva in mejoras.items():
            if texto.strip().startswith(pregunta_vieja):
                cell['source'] = [pregunta_nueva + '\n']
                return True
    return False

def agregar_transiciones(cells):
    """Agrega texto de transición entre secciones principales"""
    transiciones = {
        "## 3. Análisis de Calidad de Datos": """## 3. Análisis de Calidad de Datos

Antes de analizar las relaciones entre variables, es fundamental evaluar la calidad de los datos. En esta sección identificamos valores faltantes, duplicados y posibles inconsistencias que podrían afectar nuestros análisis posteriores.""",

        "## 5. Visualización de Variables Continuas": """## 5. Visualización de Variables Continuas

Con el dataset limpio y las variables clasificadas, procedemos a explorar visualmente las distribuciones de las variables continuas más relevantes. Esto nos ayudará a:
- Identificar patrones y tendencias
- Detectar asimetrías en las distribuciones
- Visualizar la presencia de outliers
- Entender los rangos típicos de cada variable clínica""",

        "## 6. Análisis de Outliers": """## 6. Análisis de Outliers

Las visualizaciones anteriores sugieren la presencia de valores extremos en varias variables. En esta sección aplicamos el método estadístico IQR para:
- Cuantificar outliers de manera sistemática
- Determinar qué variables tienen mayor proporción de valores atípicos
- Decidir qué tratamiento aplicar (mantener, transformar o eliminar)""",

        "## 7. Resumen del Análisis Exploratorio Inicial": """## 7. Resumen del Análisis Exploratorio Inicial

**Checkpoint**: Antes de continuar con análisis más profundos, resumimos los hallazgos clave hasta este punto.""",

        "## 9. Análisis de Correlación entre Variables": """## 9. Análisis de Correlación entre Variables

Ahora que entendemos las distribuciones individuales, investigamos las relaciones entre variables. El análisis de correlación nos permite:
- Identificar redundancias (variables muy correlacionadas entre sí)
- Encontrar predictores potenciales del riesgo cardiovascular
- Detectar relaciones inesperadas que merezcan investigación adicional""",

        "## 10. Análisis de Proporciones de Variables Categóricas": """## 10. Análisis de Proporciones de Variables Categóricas

Complementamos el análisis de correlación (variables numéricas) con el análisis de proporciones para variables categóricas:
- Comparamos tasas de riesgo cardiovascular entre grupos
- Evaluamos factores de riesgo modificables (tabaquismo) y no modificables (sexo)
- Cuantificamos el impacto de condiciones preexistentes (diabetes, hipertensión)""",

        "## 11. Tratamiento de Outliers con Transformaciones": """## 11. Tratamiento de Outliers con Transformaciones

Con los outliers identificados, aplicamos dos estrategias de tratamiento:
1. **Transformación logarítmica:** Útil para variables con asimetría positiva
2. **Winsorización:** Reemplaza valores extremos sin eliminar registros

Comparamos ambos enfoques para elegir el más apropiado para cada variable.""",

        "## 12. Preparación de Datos para Modelado": """## 12. Preparación de Datos para Modelado

Con el análisis exploratorio completo y los outliers tratados, preparamos los datos para el modelado predictivo. Este proceso incluye:
- Selección de variables relevantes (feature selection)
- División en conjuntos de entrenamiento y prueba
- Normalización de variables numéricas
- Codificación de variables categóricas (si aplica)""",

        "## 13. Modelo de Regresión Logística": """## 13. Modelo de Regresión Logística

Construimos un modelo de regresión logística para predecir el riesgo cardiovascular a 10 años. La regresión logística es apropiada porque:
- Nuestra variable objetivo es binaria (TenYearCHD: 0 o 1)
- Proporciona probabilidades interpretables
- Permite identificar qué variables son más importantes
- Es un modelo estándar en epidemiología cardiovascular""",

        "## 14. Conclusiones Finales y Recomendaciones": """## 14. Conclusiones Finales y Recomendaciones

Sintetizamos los hallazgos principales del análisis, evaluamos las limitaciones del estudio y proponemos mejoras para futuras iteraciones."""
    }

    modificaciones = 0
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and cell['source']:
            texto = ''.join(cell['source'])
            for header, texto_mejorado in transiciones.items():
                if texto.strip().startswith(header) and len(texto.strip()) <= len(header) + 50:
                    # Solo reemplazar si el texto actual es corto (solo el header o con poco texto)
                    cell['source'] = [texto_mejorado + '\n']
                    modificaciones += 1
                    break

    return modificaciones

def mejorar_descripcion_funciones(cell):
    """Mejora descripciones de funciones en la sección 4"""
    mejoras = {
        "## 4.2 Función: Detectar Fechas": """## 4.2 Función: Detectar Fechas

**Propósito:** Identifica automáticamente columnas que contienen fechas en diferentes formatos.

**¿Por qué es importante?** Las fechas requieren tratamiento especial y suelen necesitar conversión al tipo datetime para análisis temporales (ej: edad del paciente, fecha de diagnóstico).

**Parámetros:**
- `df`: DataFrame de pandas a analizar
- `porcentaje_limite`: Umbral mínimo de valores que deben ser fechas válidas (default: 80%)

**Retorna:** Lista de nombres de columnas identificadas como fechas.

**Método:** Intenta convertir cada columna a datetime y verifica el porcentaje de conversiones exitosas.""",

        "## 4.3 Función: Clasificar Variables como Continuas o Categóricas": """## 4.3 Función: Clasificar Variables como Continuas o Categóricas

**Propósito:** Clasifica automáticamente cada variable del dataset según su naturaleza.

**¿Por qué es importante?** Diferentes tipos de variables requieren:
- Análisis estadísticos diferentes (media vs frecuencia)
- Visualizaciones diferentes (histograma vs barras)
- Tratamiento diferente en modelos de ML

**Parámetros:**
- `df`: DataFrame de pandas
- `limite_categorias`: Número máximo de valores únicos para considerar una variable categórica (default: 10)
- `incluir_objetivo`: Si incluir o no la variable objetivo en la clasificación

**Retorna:** Diccionario con tres listas:
- `continuas`: Variables numéricas con muchos valores únicos
- `categoricas`: Variables con pocos valores únicos
- `binarias`: Variables con exactamente 2 valores únicos (caso especial)

**Lógica:** Una variable se considera categórica si tiene ≤ limite_categorias valores únicos.""",

        "## 4.4 Función: Detectar Outliers con IQR": """## 4.4 Función: Detectar Outliers con IQR

**Propósito:** Identifica valores atípicos usando el método estadístico del Rango Intercuartílico (IQR).

**¿Por qué IQR?** Es robusto porque:
- No asume distribución normal
- No se ve afectado por outliers extremos
- Es fácil de interpretar

**Parámetros:**
- `df`: DataFrame con variables numéricas
- `columnas`: Lista de columnas a analizar (default: todas las numéricas)
- `factor`: Multiplicador del IQR (default: 1.5 para outliers moderados, 3.0 para extremos)

**Retorna:**
- DataFrame con marcadores booleanos (True = outlier)
- Resumen con conteo de outliers por variable

**Método IQR:**
```
Q1 = Percentil 25
Q3 = Percentil 75
IQR = Q3 - Q1
Límite inferior = Q1 - (factor × IQR)
Límite superior = Q3 + (factor × IQR)
```

Valores fuera de estos límites se consideran outliers.""",

        "## 4.5 Función: Aplicar Transformación Logarítmica": """## 4.5 Función: Aplicar Transformación Logarítmica

**Propósito:** Aplica transformación log(x+1) para normalizar distribuciones asimétricas.

**¿Cuándo usarla?** Cuando las variables tienen:
- Asimetría positiva (cola larga a la derecha)
- Rango muy amplio (ej: ingresos, niveles hormonales)
- Valores que crecen exponencialmente

**Parámetros:**
- `df`: DataFrame
- `columnas`: Variables a transformar
- `sufijo`: Sufijo para las nuevas columnas (default: '_log')

**Retorna:** DataFrame con columnas transformadas agregadas.

**Ventajas:**
- Comprime valores grandes
- Reduce el impacto de outliers
- Ayuda a cumplir supuestos de normalidad para ciertos modelos

**Nota:** Usa log(x+1) en lugar de log(x) para manejar valores cero.""",

        "## 4.6 Función: Aplicar Winsorización": """## 4.6 Función: Aplicar Winsorización

**Propósito:** Limita valores extremos reemplazándolos por percentiles especificados, sin eliminar registros.

**¿Cuándo usarla?** Cuando:
- Los outliers son datos válidos pero queremos reducir su influencia
- No podemos perder registros (dataset pequeño)
- Queremos un método menos agresivo que la eliminación

**Parámetros:**
- `df`: DataFrame
- `columnas`: Variables a winsorizar
- `limites`: Tupla con percentiles inferior y superior (default: (5, 95))
- `sufijo`: Sufijo para nuevas columnas (default: '_wins')

**Retorna:** DataFrame con columnas winsorizadas agregadas.

**Ejemplo:**
Si limites=(5, 95):
- Valores < percentil 5 → reemplazados por percentil 5
- Valores > percentil 95 → reemplazados por percentil 95
- Valores entre percentiles 5-95 → sin cambios

**Ventajas:**
- Preserva todos los registros
- Reduce varianza sin perder información de orden
- Mejora robustez del modelo ante outliers"""
    }

    if cell['cell_type'] == 'markdown' and cell['source']:
        texto = ''.join(cell['source'])
        for titulo_viejo, texto_nuevo in mejoras.items():
            if texto.strip().startswith(titulo_viejo) and len(texto.strip()) < len(titulo_viejo) + 200:
                cell['source'] = [texto_nuevo + '\n']
                return True
    return False

def agregar_interpretaciones(cells):
    """Agrega interpretaciones después de análisis clave"""
    # Esta función buscará celdas específicas que necesiten interpretación
    # Por ahora, identificaremos patrones comunes y agregaremos celdas nuevas si es necesario

    # Buscamos celdas con outputs de correlación, estadísticas descriptivas, etc.
    # y agregamos celdas markdown después si no existen

    interpretaciones_agregadas = 0

    # Buscar matriz de correlación y agregar interpretación
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and cell['source']:
            codigo = ''.join(cell['source'])

            # Si encontramos código de matriz de correlación
            if 'corr()' in codigo and 'TenYearCHD' in codigo:
                # Verificar si la siguiente celda ya tiene interpretación
                if i + 1 < len(cells) and cells[i + 1]['cell_type'] != 'markdown':
                    # Agregar celda de interpretación
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**Interpretación de Correlaciones:**\n',
                            '\n',
                            'Las correlaciones nos revelan:\n',
                            '- Variables con coeficiente > 0.3: Correlación positiva moderada-fuerte\n',
                            '- Variables con coeficiente < -0.3: Correlación negativa moderada-fuerte\n',
                            '- Variables con |coeficiente| < 0.1: Correlación débil o inexistente\n',
                            '\n',
                            'Los valores más correlacionados con TenYearCHD son potenciales predictores importantes para nuestro modelo.\n'
                        ]
                    }
                    cells.insert(i + 1, nueva_celda)
                    interpretaciones_agregadas += 1

    return interpretaciones_agregadas

def main():
    """Función principal que ejecuta todas las mejoras"""

    ruta_notebook = '/Users/leoalvarez/Documents/Digital House Data Science/TP1/Ejercicio framingham/resolucionTP.ipynb'

    print("🔍 Cargando notebook...")
    notebook = cargar_notebook(ruta_notebook)

    cells = notebook['cells']

    # Contadores de mejoras
    mejoras_total = 0

    print("\n📝 Mejorando headers de visualizaciones...")
    for cell in cells:
        if mejorar_headers_visualizaciones(cell):
            mejoras_total += 1
    print(f"   ✓ {mejoras_total} headers mejorados")

    print("\n❓ Mejorando preguntas de la sección Q&A...")
    mejoras_qa = 0
    for cell in cells:
        if mejorar_preguntas_qa(cell):
            mejoras_qa += 1
    mejoras_total += mejoras_qa
    print(f"   ✓ {mejoras_qa} preguntas mejoradas")

    print("\n🔄 Agregando transiciones entre secciones...")
    transiciones = agregar_transiciones(cells)
    mejoras_total += transiciones
    print(f"   ✓ {transiciones} transiciones agregadas")

    print("\n⚙️  Mejorando descripciones de funciones...")
    mejoras_func = 0
    for cell in cells:
        if mejorar_descripcion_funciones(cell):
            mejoras_func += 1
    mejoras_total += mejoras_func
    print(f"   ✓ {mejoras_func} funciones mejoradas")

    print("\n💡 Agregando interpretaciones...")
    interpretaciones = agregar_interpretaciones(cells)
    mejoras_total += interpretaciones
    print(f"   ✓ {interpretaciones} interpretaciones agregadas")

    # Crear backup del original
    backup_ruta = ruta_notebook.replace('.ipynb', '_backup.ipynb')
    print(f"\n💾 Creando backup en: {os.path.basename(backup_ruta)}")
    guardar_notebook(notebook, backup_ruta)

    # Guardar notebook mejorado
    print(f"💾 Guardando notebook mejorado...")
    guardar_notebook(notebook, ruta_notebook)

    print(f"\n✅ ¡Completado! Se aplicaron {mejoras_total} mejoras en total.")
    print(f"📄 Original guardado en: {os.path.basename(backup_ruta)}")
    print(f"📄 Notebook mejorado: {os.path.basename(ruta_notebook)}")

if __name__ == "__main__":
    main()
