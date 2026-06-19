#!/usr/bin/env python3
"""
Versión 3: Mejoras finales de contexto y formato
"""

import json
import re

def cargar_notebook(ruta):
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)

def guardar_notebook(notebook, ruta):
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def mejorar_titulos_principales(cell):
    """Mejora títulos de secciones principales que están muy escuetos"""

    if cell['cell_type'] != 'markdown' or not cell['source']:
        return False

    texto = ''.join(cell['source']).strip()

    # Diccionario de mejoras para títulos principales
    mejoras = {
        # Título principal si es muy simple
        "# Análisis del Dataset Framingham": """# 🫀 Análisis del Dataset Framingham Heart Study

## Predicción de Riesgo de Enfermedad Cardiovascular a 10 Años

**Contexto del Estudio:**
El Framingham Heart Study, iniciado en 1948, es uno de los estudios epidemiológicos más influyentes en la historia de la cardiología. Este análisis trabaja con datos de pacientes evaluados para múltiples factores de riesgo cardiovascular.

**Objetivos de Este Análisis:**
1. **Exploración:** Comprender las características y distribución de las variables
2. **Limpieza:** Identificar y tratar valores faltantes y outliers
3. **Análisis:** Descubrir relaciones entre factores de riesgo y enfermedad cardiovascular
4. **Modelado:** Desarrollar un modelo predictivo para estimar riesgo a 10 años
5. **Interpretación:** Extraer insights accionables para prevención

**Variables del Dataset:**
- **Demográficas:** Edad, sexo
- **Comportamentales:** Tabaquismo, actividad física
- **Clínicas:** Presión arterial (sistólica/diastólica), colesterol, glucosa, IMC, frecuencia cardíaca
- **Historial Médico:** Hipertensión previa, diabetes, uso de medicación
- **Variable Objetivo:** `TenYearCHD` - Desarrollo de enfermedad coronaria en próximos 10 años (1=Sí, 0=No)""",

        "## 1. Preparación y Carga de Datos": """## 1. Preparación y Carga de Datos

**Objetivo:** Configurar el entorno de análisis y cargar el dataset.

En esta sección:
- Importamos las librerías necesarias (pandas, numpy, matplotlib, seaborn, sklearn)
- Configuramos opciones de visualización
- Cargamos el archivo CSV del estudio Framingham
- Verificamos la carga correcta con una primera inspección""",

        "## 2. Inspección Inicial del Dataset": """## 2. Inspección Inicial del Dataset

**Objetivo:** Realizar una primera exploración para entender la estructura y calidad de los datos.

Respondemos preguntas iniciales:
- ¿Cuántos pacientes y variables tenemos?
- ¿Qué tipos de datos contiene cada columna?
- ¿Cómo lucen los primeros registros?
- ¿Hay valores faltantes evidentes?

Esta inspección inicial guía todas las decisiones posteriores de limpieza y análisis.""",

        "## 4. Identificación de Tipos de Variables": """## 4. Identificación de Tipos de Variables

**Objetivo:** Clasificar automáticamente cada variable según su naturaleza estadística.

**¿Por qué es importante?**
- Las **variables continuas** (edad, presión arterial) requieren análisis descriptivos, histogramas, correlaciones
- Las **variables categóricas** (sexo, fumador) requieren frecuencias, proporciones, tablas de contingencia
- Los **modelos de ML** tratan cada tipo de forma diferente

Desarrollamos funciones personalizadas para automatizar esta clasificación y hacerla reproducible.""",

        "## 8. Respuestas a Preguntas del Dataset": """## 8. Respuestas a Preguntas del Dataset

**Objetivo:** Consolidar los hallazgos del análisis exploratorio respondiendo preguntas clave sobre los datos.

Esta sección sintetiza la información más importante descubierta hasta ahora y sirve como referencia rápida para decisiones de modelado. Cada pregunta aborda un aspecto fundamental: tamaño de la muestra, tipos de variables, calidad de datos, distribuciones, outliers, correlaciones y efectos de transformaciones."""
    }

    for patron, mejora in mejoras.items():
        # Buscar si el texto comienza con este patrón y es corto
        if texto.startswith(patron):
            # Si el texto actual tiene menos de 200 caracteres adicionales al título, mejorarlo
            if len(texto) < len(patron) + 200:
                cell['source'] = [mejora + '\n']
                return True

    return False

def mejorar_preguntas_restantes(cell):
    """Mejora las preguntas que aún no tienen contexto suficiente"""

    if cell['cell_type'] != 'markdown' or not cell['source']:
        return False

    texto = ''.join(cell['source']).strip()

    # Buscar preguntas que sean muy cortas (solo el título de la pregunta)
    if texto.startswith('## 8.') and len(texto) < 100:
        # Estas son preguntas cortas que necesitan contexto
        if '¿Cuántos registros hay?' in texto:
            cell['source'] = ["""## 8.1 ¿Cuántos registros hay?

**Propósito:** Cuantificar el tamaño de la muestra para evaluar la robustez estadística del análisis.

Un dataset más grande permite:
- Mayor poder estadístico para detectar efectos
- Mejor generalización del modelo
- Particiones train/test más balanceadas

El número de variables indica la complejidad del análisis y el riesgo de overfitting (regla general: se necesitan al menos 10-20 observaciones por variable predictora).\n"""]
            return True

        elif '¿Qué tipo de variable es cada una?' in texto:
            cell['source'] = ["""## 8.2 ¿Qué tipo de variable es cada una?

**Propósito:** Clasificar cada variable según su naturaleza estadística y computacional.

**Tipos relevantes:**
- **Numéricas continuas:** Valores en rango continuo (edad, presión arterial, colesterol)
- **Numéricas discretas:** Conteos o categorías numéricas (cigarrillos por día)
- **Categóricas nominales:** Sin orden (sexo, fumador sí/no)
- **Categóricas ordinales:** Con orden (nivel educativo, estadio de enfermedad)

Esta clasificación determina:
- Qué visualizaciones usar (histograma vs barras)
- Qué estadísticos calcular (media vs frecuencia)
- Cómo preprocesar para modelado (normalización vs one-hot encoding)\n"""]
            return True

        elif '¿Hay valores faltantes?' in texto:
            cell['source'] = ["""## 8.3 ¿Hay valores faltantes?

**Propósito:** Identificar la completitud de los datos y planificar estrategias de imputación.

**¿Por qué faltan datos?**
- **MCAR (Missing Completely At Random):** Falta aleatoriamente - seguro para eliminar
- **MAR (Missing At Random):** Depende de otras variables observadas - imputación recomendada
- **MNAR (Missing Not At Random):** El hecho de faltar es informativo - tratamiento especial

**Estrategias según % faltante:**
- **< 5%:** Eliminar registros (pérdida mínima)
- **5-20%:** Imputación (media, mediana, moda, KNN)
- **> 20%:** Considerar eliminar variable o modelado predictivo

Los valores faltantes en variables predictoras clave pueden sesgar resultados.\n"""]
            return True

        elif '¿Cómo se distribuyen las variables continuas principales?' in texto:
            cell['source'] = ["""## 8.4 ¿Cómo se distribuyen las variables continuas principales?

**Propósito:** Analizar forma, centralidad y dispersión de variables numéricas.

**Aspectos a evaluar:**
- **Tendencia central:** Media vs mediana (si difieren mucho → asimetría)
- **Dispersión:** Desviación estándar y rango (alta variabilidad → considerar normalización)
- **Forma:** Simetría, curtosis, modalidad
- **Rango:** Valores mínimo y máximo (detectar posibles errores)

**Implicaciones para modelado:**
- Distribuciones muy asimétricas pueden beneficiarse de transformaciones (log, raíz cuadrada)
- Rangos muy diferentes entre variables sugieren necesidad de escalado
- Distribuciones bimodales pueden indicar subpoblaciones ocultas\n"""]
            return True

        elif '¿Existen outliers en las variables continuas?' in texto:
            cell['source'] = ["""## 8.5 ¿Existen outliers en las variables continuas?

**Propósito:** Detectar valores extremos que podrían afectar el análisis y modelado.

**Tipos de outliers:**
- **Válidos:** Valores reales pero inusuales (ej: paciente con presión arterial 200 mmHg)
- **Errores:** Datos mal registrados (ej: edad = 999)
- **Influyentes:** Impactan desproporcionadamente en estadísticos y modelos

**Método IQR (Interquartile Range):**
- Robusto - no asume distribución normal
- Límite inferior: Q1 - 1.5×IQR
- Límite superior: Q3 + 1.5×IQR
- Valores fuera de estos límites = outliers

**Tratamiento:**
- **Eliminar:** Si son errores claros
- **Transformar:** Aplicar log o winsorización
- **Mantener:** Si son casos válidos importantes (ej: pacientes de alto riesgo)\n"""]
            return True

        elif '¿Qué variables están correlacionadas con el riesgo cardiovascular' in texto:
            cell['source'] = ["""## 8.6 ¿Qué variables están correlacionadas con el riesgo cardiovascular (TenYearCHD)?

**Propósito:** Identificar los factores más asociados con el desarrollo de enfermedad cardiovascular.

**Interpretación de coeficientes de correlación (r de Pearson):**
- **|r| < 0.1:** Correlación débil/inexistente
- **0.1 ≤ |r| < 0.3:** Correlación débil
- **0.3 ≤ |r| < 0.7:** Correlación moderada
- **|r| ≥ 0.7:** Correlación fuerte
- **r > 0:** Relación positiva (aumenta riesgo)
- **r < 0:** Relación negativa (disminuye riesgo)

**Factores esperados según literatura médica:**
- **Positivos:** Edad, presión arterial, colesterol, diabetes, tabaquismo
- **Negativos:** (raros en este contexto)

**Nota importante:** Correlación ≠ Causalidad. Una correlación fuerte sugiere asociación, pero no prueba que una variable cause la otra.\n"""]
            return True

        elif '¿Las variables categóricas muestran diferencias en la proporción de riesgo?' in texto:
            cell['source'] = ["""## 8.7 ¿Las variables categóricas muestran diferencias en la proporción de riesgo?

**Propósito:** Comparar tasas de eventos cardiovasculares entre diferentes grupos.

**Variables categóricas clave:**
- **Sexo (male):** ¿Hombres tienen mayor riesgo que mujeres?
- **Fumador (currentSmoker):** ¿El tabaquismo aumenta el riesgo?
- **Medicación (BPMeds):** ¿Pacientes medicados tienen perfil diferente?
- **Hipertensión (prevalentHyp):** ¿Cuánto aumenta el riesgo?
- **Diabetes (diabetes):** ¿Cuál es el impacto de la diabetes?

**Análisis:**
- Calculamos porcentaje de TenYearCHD=1 en cada grupo
- Comparamos proporciones (ej: % riesgo en fumadores vs no fumadores)
- Diferencias > 5-10% son clínicamente relevantes

Estos factores categóricos son útiles para:
- Segmentación de pacientes
- Identificación de grupos de alto riesgo
- Desarrollo de programas de prevención dirigidos\n"""]
            return True

        elif '¿Cómo afectan las transformaciones a la distribución de outliers?' in texto:
            cell['source'] = ["""## 8.8 ¿Cómo afectan las transformaciones a la distribución de outliers?

**Propósito:** Evaluar la efectividad de dos técnicas para reducir el impacto de valores extremos.

**Técnica 1: Transformación Logarítmica (log)**
- **Método:** Aplicar log(x+1) a cada valor
- **Efecto:** Comprime valores grandes, expande valores pequeños
- **Ventajas:**
  - Reduce asimetría positiva
  - Normaliza distribuciones sesgadas
  - Interpretable (cambios multiplicativos → cambios aditivos)
- **Desventajas:**
  - Cambia la escala original (dificulta interpretación clínica)
  - No funciona con valores negativos

**Técnica 2: Winsorización**
- **Método:** Reemplazar outliers por percentiles límite (ej: 5° y 95°)
- **Efecto:** Limita valores extremos sin eliminarlos
- **Ventajas:**
  - Mantiene el número de registros
  - Conserva escala original
  - Reduce varianza
- **Desventajas:**
  - Pierde información de valores extremos verdaderos
  - Arbitrario (¿qué percentiles usar?)

**Comparación:** Visualizamos distribuciones antes/después para decidir qué método es más apropiado para cada variable.\n"""]
            return True

        elif '¿Qué rendimiento obtiene el modelo de regresión logística?' in texto:
            cell['source'] = ["""## 8.9 ¿Qué rendimiento obtiene el modelo de regresión logística?

**Propósito:** Evaluar la capacidad predictiva del modelo usando múltiples métricas.

**Métricas de Evaluación:**

**1. Accuracy (Exactitud):**
- Porcentaje total de predicciones correctas
- **Limitación:** Engañosa con clases desbalanceadas (puede ser alta prediciendo siempre la clase mayoritaria)

**2. Precision (Precisión):**
- De los pacientes que predecimos con riesgo, ¿cuántos realmente lo tienen?
- `Precision = TP / (TP + FP)`
- **Alta precisión → Pocas falsas alarmas**

**3. Recall (Sensibilidad):**
- De los pacientes con riesgo real, ¿a cuántos detectamos?
- `Recall = TP / (TP + FN)`
- **Alto recall → Detectamos la mayoría de casos**
- **Crucial en medicina:** No queremos dejar pasar pacientes en riesgo

**4. F1-Score:**
- Media armónica entre precision y recall
- `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Útil con clases desbalanceadas**

**5. ROC-AUC (Area Under the Curve):**
- Mide capacidad de discriminación del modelo
- **0.5:** Modelo aleatorio (inútil)
- **0.7-0.8:** Discriminación aceptable
- **0.8-0.9:** Discriminación excelente
- **> 0.9:** Discriminación excepcional

**Trade-off médico:** En predicción de riesgo cardiovascular, preferimos **alto recall** (detectar todos los casos) aunque tengamos más falsos positivos (mejor prevenir de más que dejar casos sin detectar).\n"""]
            return True

    return False

def agregar_conclusiones_seccionales(cells):
    """Agrega pequeñas conclusiones al final de secciones mayores"""

    nuevas_celdas = []

    # Buscar finales de sección para agregar mini-conclusiones
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and cell['source']:
            texto = ''.join(cell['source']).strip()

            # Después de la sección de limpieza de datos (sección 3)
            if 'Eliminados' in texto and 'registros con valores faltantes' in texto:
                if i + 1 < len(cells) and not (cells[i + 1]['cell_type'] == 'markdown' and 'Checkpoint' in ''.join(cells[i + 1]['source'])):
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '---\n',
                            '**✅ Checkpoint - Limpieza de Datos Completada:**\n',
                            '- Dataset limpio sin valores faltantes\n',
                            '- Tamaño de muestra preservado/ajustado\n',
                            '- Listo para análisis exploratorio en profundidad\n',
                            '\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))

    # Insertar nuevas celdas
    for offset, (indice, celda) in enumerate(sorted(nuevas_celdas)):
        cells.insert(indice + offset, celda)

    return len(nuevas_celdas)

def main():
    ruta_notebook = '/Users/leoalvarez/Documents/Digital House Data Science/TP1/Ejercicio framingham/resolucionTP.ipynb'

    print("🔍 Cargando notebook...")
    notebook = cargar_notebook(ruta_notebook)
    cells = notebook['cells']

    mejoras_total = 0

    print("\n📝 Mejorando títulos principales...")
    mejoras_titulos = 0
    for cell in cells:
        if mejorar_titulos_principales(cell):
            mejoras_titulos += 1
    mejoras_total += mejoras_titulos
    print(f"   ✓ {mejoras_titulos} títulos mejorados")

    print("\n❓ Mejorando preguntas restantes...")
    mejoras_preguntas = 0
    for cell in cells:
        if mejorar_preguntas_restantes(cell):
            mejoras_preguntas += 1
    mejoras_total += mejoras_preguntas
    print(f"   ✓ {mejoras_preguntas} preguntas mejoradas")

    print("\n🎯 Agregando conclusiones seccionales...")
    conclusiones = agregar_conclusiones_seccionales(cells)
    mejoras_total += conclusiones
    print(f"   ✓ {conclusiones} conclusiones agregadas")

    print(f"\n💾 Guardando notebook mejorado...")
    guardar_notebook(notebook, ruta_notebook)

    print(f"\n✅ ¡Completado! Se aplicaron {mejoras_total} mejoras adicionales.")
    print(f"📊 Total de celdas ahora: {len(cells)}")
    print("\n" + "="*60)
    print("📈 RESUMEN TOTAL DE MEJORAS APLICADAS:")
    print("="*60)
    print("Script v1: Headers visualizaciones, preguntas Q&A, funciones")
    print("Script v2: Celdas explicativas después de análisis")
    print("Script v3: Títulos principales, preguntas restantes, conclusiones")
    print("="*60)

if __name__ == "__main__":
    main()
