# Datasets — Módulo 01

Datasets usados en los ejercicios y prácticas del módulo.

| Archivo | Descripción | Target típico | Tipo de problema |
|---------|-------------|---------------|------------------|
| `abalone.csv` | Medidas físicas de abalones | `Rings` (edad) | Regresión |
| `breast-cancer.csv` | Diagnóstico de cáncer de mama (Wisconsin) | `diagnosis` (M/B) | Clasificación |
| `Default.csv` | Impago de tarjeta de crédito (ISLR) | `default` | Clasificación |
| `diabetes.csv` | Indicadores clínicos (Pima) | `Outcome` | Clasificación |
| `framingham.csv` | Estudio cardiovascular Framingham | `TenYearCHD` | Clasificación |
| `HR_comma_sep.csv` | Rotación de empleados (HR analytics) | `left` | Clasificación |
| `mall_customers.csv` | Clientes de shopping (segmentación) | — (sin label) | Clustering |
| `Movie_classification.csv` | Datos de películas | `Start_Tech_Oscar` | Clasificación / Árboles |

> Datasets livianos versionados directamente. Los pesados (`*.zip` > 100 MB) no se versionan: ver [`scripts/descargar_datos.sh`](../../scripts/descargar_datos.sh).
