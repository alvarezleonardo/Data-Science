#!/usr/bin/env python3
"""
Script mejorado para agregar más contexto y explicaciones a la notebook
Versión 2: Más agresivo con las mejoras
"""

import json
import re

def cargar_notebook(ruta):
    """Carga el notebook JSON"""
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)

def guardar_notebook(notebook, ruta):
    """Guarda el notebook JSON con formato"""
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def agregar_celdas_explicativas(cells):
    """Agrega celdas de explicación después de análisis importantes"""

    nuevas_celdas = []
    indices_procesados = set()

    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and cell['source']:
            codigo = ''.join(cell['source'])

            # Después de df.describe()
            if 'describe()' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**📊 Interpretación de Estadísticas Descriptivas:**\n',
                            '\n',
                            'Las estadísticas básicas nos revelan:\n',
                            '- **count:** Número de valores no nulos (ayuda a identificar missing values)\n',
                            '- **mean vs median:** Si difieren mucho, indica asimetría en la distribución\n',
                            '- **std:** Desviación estándar - alta variabilidad indica mayor dispersión\n',
                            '- **min/max:** Detecta posibles errores de entrada o valores extremos\n',
                            '- **percentiles (25%, 50%, 75%):** Muestran la distribución de los datos\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de info()
            elif 'info()' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**🔍 Información Clave del Dataset:**\n',
                            '\n',
                            'Esta información nos permite:\n',
                            '- Confirmar los tipos de datos de cada variable\n',
                            '- Identificar el uso de memoria del dataset\n',
                            '- Detectar columnas con valores nulos (Non-Null Count < Total)\n',
                            '- Planificar conversiones de tipo si son necesarias\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de isnull().sum()
            elif 'isnull().sum()' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**🔴 Estrategias para Manejar Valores Faltantes:**\n',
                            '\n',
                            'Dependiendo del porcentaje de missing values, podemos:\n',
                            '- **< 5%:** Eliminar registros (pérdida mínima de información)\n',
                            '- **5-20%:** Imputación con media/mediana (continuas) o moda (categóricas)\n',
                            '- **> 20%:** Considerar eliminar la variable o usar técnicas avanzadas (KNN, modelos predictivos)\n',
                            '\n',
                            'También es importante entender **por qué** faltan los datos (MCAR, MAR, MNAR) antes de decidir la estrategia.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de value_counts() de TenYearCHD
            elif 'TenYearCHD' in codigo and 'value_counts()' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**⚖️ Balance de Clases:**\n',
                            '\n',
                            'La variable objetivo (TenYearCHD) es **desbalanceada**. Esto es común en problemas médicos donde:\n',
                            '- La clase negativa (0 = sin evento) es mayoritaria\n',
                            '- La clase positiva (1 = con evento cardiovascular) es minoritaria\n',
                            '\n',
                            'Este desbalance puede afectar el modelo:\n',
                            '- **Problema:** El modelo puede aprender a predecir siempre la clase mayoritaria\n',
                            '- **Solución:** Usar métricas apropiadas (Precision, Recall, F1, ROC-AUC) en lugar de solo Accuracy\n',
                            '- **Alternativas:** SMOTE, class_weight, threshold tuning\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de boxplot o violinplot
            elif ('boxplot' in codigo or 'violinplot' in codigo) and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**📦 Lectura de Boxplots:**\n',
                            '\n',
                            'Elementos clave del boxplot:\n',
                            '- **Caja:** Contiene el 50% central de los datos (IQR)\n',
                            '- **Línea central:** Mediana (percentil 50)\n',
                            '- **Bigotes:** Se extienden hasta 1.5×IQR desde los bordes de la caja\n',
                            '- **Puntos fuera:** Outliers potenciales\n',
                            '\n',
                            'Una caja simétrica indica distribución normal, mientras que asimetría sugiere sesgo positivo o negativo.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)
                    break  # Solo agregar una vez

            # Después de heatmap de correlación
            elif 'heatmap' in codigo and 'corr()' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**🔥 Interpretación del Heatmap de Correlación:**\n',
                            '\n',
                            'Colores e interpretación:\n',
                            '- **Rojo intenso (cercano a 1):** Correlación positiva fuerte - variables aumentan juntas\n',
                            '- **Azul intenso (cercano a -1):** Correlación negativa fuerte - cuando una sube, la otra baja\n',
                            '- **Blanco/neutro (cercano a 0):** Sin correlación lineal\n',
                            '\n',
                            'Puntos importantes:\n',
                            '- Diagonal siempre es 1 (variable consigo misma)\n',
                            '- Multicolinealidad: Variables predictoras muy correlacionadas entre sí (> 0.8) pueden causar problemas\n',
                            '- Buscar correlaciones con TenYearCHD para identificar predictores potenciales\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de StandardScaler
            elif 'StandardScaler' in codigo and 'fit_transform' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**⚙️ ¿Por qué Normalizar los Datos?**\n',
                            '\n',
                            'El StandardScaler transforma las variables para que tengan:\n',
                            '- **Media = 0**\n',
                            '- **Desviación estándar = 1**\n',
                            '\n',
                            'Esto es importante porque:\n',
                            '- Variables con diferentes escalas (ej: edad 20-80 vs glucosa 50-400) pueden dominar el modelo\n',
                            '- Algoritmos basados en distancias (regresión logística, KNN, SVM) son sensibles a la escala\n',
                            '- Mejora la convergencia de algoritmos de optimización (gradient descent)\n',
                            '\n',
                            '⚠️ **Importante:** Solo ajustamos (fit) con datos de entrenamiento para evitar data leakage.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de train_test_split
            elif 'train_test_split' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**✂️ División Train-Test:**\n',
                            '\n',
                            'Separamos los datos para:\n',
                            '- **Entrenamiento (80%):** El modelo aprende patrones de estos datos\n',
                            '- **Prueba (20%):** Evaluamos el rendimiento en datos nunca vistos\n',
                            '\n',
                            'Esto simula cómo el modelo funcionará con pacientes nuevos en el mundo real.\n',
                            '\n',
                            '**random_state:** Fija la semilla para reproducibilidad - obtendremos siempre la misma división.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de LogisticRegression
            elif 'LogisticRegression' in codigo and 'fit(' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**🎯 Modelo Entrenado:**\n',
                            '\n',
                            'El modelo ha aprendido los pesos (coeficientes) para cada variable que minimizan el error de predicción.\n',
                            '\n',
                            'La regresión logística calcula la probabilidad de riesgo cardiovascular usando:\n',
                            '```\n',
                            'P(TenYearCHD=1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))\n',
                            '```\n',
                            '\n',
                            'Donde cada β representa cuánto influye cada variable en el riesgo.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de classification_report
            elif 'classification_report' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**📈 Métricas de Evaluación:**\n',
                            '\n',
                            '- **Precision (Precisión):** De todos los que predijimos con riesgo, ¿cuántos realmente lo tienen?\n',
                            '  - Alta precision = pocas falsas alarmas\n',
                            '\n',
                            '- **Recall (Sensibilidad):** De todos los que tienen riesgo, ¿a cuántos detectamos?\n',
                            '  - Alto recall = detectamos la mayoría de casos reales\n',
                            '\n',
                            '- **F1-Score:** Media armónica entre precision y recall\n',
                            '  - Útil cuando hay desbalance de clases\n',
                            '\n',
                            '- **Support:** Número de casos reales en cada clase\n',
                            '\n',
                            '**Trade-off:** En medicina, preferimos alto recall (detectar todos los casos de riesgo) aunque tengamos más falsos positivos.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de confusion_matrix
            elif 'confusion_matrix' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**🎲 Interpretación de la Matriz de Confusión:**\n',
                            '\n',
                            '```\n',
                            '                Predicho\n',
                            '              0         1\n',
                            'Real   0    TN        FP\n',
                            '       1    FN        TP\n',
                            '```\n',
                            '\n',
                            '- **TN (True Negatives):** Correctamente predecimos sin riesgo ✅\n',
                            '- **TP (True Positives):** Correctamente predecimos con riesgo ✅\n',
                            '- **FP (False Positives):** Error tipo I - predecimos riesgo pero no lo hay ⚠️\n',
                            '- **FN (False Negatives):** Error tipo II - no detectamos el riesgo real ❌ (más grave en medicina)\n',
                            '\n',
                            'En contexto médico: **Los FN son más peligrosos** porque dejamos pasar pacientes en riesgo sin tratamiento.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

            # Después de roc_auc_score
            elif 'roc_auc_score' in codigo and i not in indices_procesados:
                if i + 1 >= len(cells) or cells[i + 1]['cell_type'] != 'markdown' or len(''.join(cells[i + 1]['source'])) < 50:
                    nueva_celda = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '**📊 ROC-AUC Score:**\n',
                            '\n',
                            'El Area Under the ROC Curve (AUC) mide la capacidad del modelo para discriminar entre clases:\n',
                            '\n',
                            '- **AUC = 0.5:** Modelo no mejor que azar (adivinar al azar)\n',
                            '- **AUC = 0.7-0.8:** Discriminación aceptable\n',
                            '- **AUC = 0.8-0.9:** Discriminación excelente\n',
                            '- **AUC > 0.9:** Discriminación excepcional\n',
                            '- **AUC = 1.0:** Clasificación perfecta (sospechoso de overfitting)\n',
                            '\n',
                            'Esta métrica es robusta al desbalance de clases y preferida en problemas médicos.\n'
                        ]
                    }
                    nuevas_celdas.append((i + 1, nueva_celda))
                    indices_procesados.add(i)

    # Insertar las nuevas celdas
    for offset, (indice, celda) in enumerate(sorted(nuevas_celdas)):
        cells.insert(indice + offset, celda)

    return len(nuevas_celdas)

def mejorar_secciones_introductorias(cell):
    """Mejora secciones que tienen solo el título sin contexto"""
    mejoras = {
        "# Análisis del Dataset Framingham Heart Study": """# 🫀 Análisis del Dataset Framingham Heart Study

## Predicción de Riesgo de Enfermedad Cardiovascular a 10 Años

**Objetivos del Análisis:**
- Explorar y comprender las características del dataset Framingham
- Identificar factores de riesgo asociados con enfermedad cardiovascular
- Desarrollar un modelo predictivo para estimar riesgo a 10 años
- Proporcionar insights accionables para prevención cardiovascular

**Sobre el Dataset:**
El Framingham Heart Study es uno de los estudios epidemiológicos más importantes en cardiología, iniciado en 1948. Este dataset contiene información de **pacientes** evaluados para factores de riesgo cardiovascular.

**Variables Clave:**
- **Demográficas:** Edad, sexo
- **Comportamentales:** Tabaquismo, ejercicio
- **Clínicas:** Presión arterial, colesterol, glucosa, IMC
- **Historial médico:** Hipertensión, diabetes, medicación
- **Objetivo:** TenYearCHD (enfermedad coronaria en 10 años)""",

        "## 1. Preparación y Carga de Datos": """## 1. Preparación y Carga de Datos

Comenzamos importando las librerías necesarias y cargando el dataset. Este paso inicial es fundamental para:
- Establecer el entorno de trabajo
- Verificar que los datos se cargan correctamente
- Hacer una primera inspección visual""",

        "## 2. Inspección Inicial del Dataset": """## 2. Inspección Inicial del Dataset

Realizamos una primera exploración para entender:
- Tamaño del dataset (filas y columnas)
- Tipos de variables
- Primeros registros (identificar formato y estructura)
- Valores faltantes iniciales

Esta inspección guía todas las decisiones posteriores de limpieza y análisis.""",

        "## 4. Identificación de Tipos de Variables": """## 4. Identificación de Tipos de Variables

Clasificar correctamente las variables es esencial porque:
- Diferentes tipos requieren diferentes análisis estadísticos
- Las visualizaciones apropiadas dependen del tipo de variable
- Los modelos de ML tratan variables numéricas y categóricas de forma diferente

Creamos funciones automáticas para clasificar variables según sus características."""
    }

    if cell['cell_type'] == 'markdown' and cell['source']:
        texto = ''.join(cell['source']).strip()
        for titulo, texto_mejorado in mejoras.items():
            # Si el texto actual es solo el título o muy corto
            if texto.startswith(titulo) and len(texto) < len(titulo) + 100:
                cell['source'] = [texto_mejorado + '\n']
                return True
    return False

def main():
    """Función principal"""
    ruta_notebook = '/Users/leoalvarez/Documents/Digital House Data Science/TP1/Ejercicio framingham/resolucionTP.ipynb'

    print("🔍 Cargando notebook...")
    notebook = cargar_notebook(ruta_notebook)
    cells = notebook['cells']

    mejoras_total = 0

    print("\n📝 Mejorando secciones introductorias...")
    mejoras_intro = 0
    for cell in cells:
        if mejorar_secciones_introductorias(cell):
            mejoras_intro += 1
    mejoras_total += mejoras_intro
    print(f"   ✓ {mejoras_intro} secciones mejoradas")

    print("\n💡 Agregando celdas explicativas después de análisis...")
    explicativas = agregar_celdas_explicativas(cells)
    mejoras_total += explicativas
    print(f"   ✓ {explicativas} celdas explicativas agregadas")

    print(f"\n💾 Guardando notebook mejorado...")
    guardar_notebook(notebook, ruta_notebook)

    print(f"\n✅ ¡Completado! Se aplicaron {mejoras_total} mejoras adicionales.")
    print(f"📊 Total de celdas ahora: {len(cells)}")

if __name__ == "__main__":
    main()
