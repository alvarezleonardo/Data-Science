# Criterios de Selección de Modelos

> Conversión a Markdown de la slide del curso (Clase 18, Módulo 4). El PDF original está al lado.

Los **criterios de selección de modelos** son herramientas clave para comparar y elegir entre diferentes modelos estadísticos. Los más comunes son el **Criterio de Información de Akaike (AIC)** y el **Criterio de Información Bayesiano (BIC)**. Ambos ayudan a seleccionar el modelo que mejor **equilibra el ajuste con la complejidad**, penalizando los modelos con demasiados parámetros (evitando el sobreajuste).

## 1. Criterio de Información de Akaike (AIC)

El **AIC** es una medida para seleccionar el modelo que mejor equilibra la **bondad del ajuste** con la **complejidad**. Fue introducido por Hirotugu Akaike en 1974.

```
AIC = 2k - 2·ln(L)
```

- **k** = número de parámetros del modelo.
- **L** = máxima verosimilitud del modelo.

**Características:**
- **Penalización por complejidad:** penaliza los modelos con más parámetros para evitar el sobreajuste.
- **Comparación relativa:** un **valor más bajo de AIC** indica un mejor equilibrio entre ajuste y simplicidad. Sirve para comparar modelos entre sí (no es una medida absoluta).

**Ejemplos de uso:**
- **Regresión:** en predicción de precios de viviendas, se comparan modelos con distintos conjuntos de características; el de **menor AIC** es preferible (buen ajuste con menos parámetros).
- **Series temporales:** en la predicción de ventas mensuales, se comparan distintos modelos **ARIMA (p, d, q)**; el AIC ayuda a determinar el que mejor equilibra ajuste y complejidad.

## 2. Criterio de Información Bayesiano (BIC)

El **BIC**, también conocido como **Criterio de Información de Schwarz (SIC)**, es otro criterio para la selección de modelos que equilibra ajuste y complejidad. Fue propuesto por Gideon E. Schwarz en 1978.

```
BIC = ln(n)·k - 2·ln(L)
```

- **n** = número de observaciones.
- **k** = número de parámetros del modelo.
- **L** = máxima verosimilitud del modelo.

**Características:**
- Penaliza **más severamente** los modelos con más parámetros que el AIC, y esa penalización **aumenta con el tamaño de la muestra** (`ln(n)` en vez de `2`).
- Igual que el AIC, un **menor valor de BIC** indica una mejor selección de modelo.

**Ejemplos de uso:**
- **Modelos de clustering:** en segmentación de clientes, se compara el BIC para distintos números de clusters en un **modelo de mezcla gaussiana (GMM)**; el número de clusters que **minimiza el BIC** es preferible.
- **Modelos de mezcla:** en la identificación de subgrupos en datos genéticos, se comparan modelos de mezcla con distinto número de componentes y se elige el que mejor ajusta **sin sobreajustar**.

## 3. AIC vs BIC

| | AIC | BIC |
|--|-----|-----|
| **Fórmula** | `2k - 2·ln(L)` | `ln(n)·k - 2·ln(L)` |
| **Penalización por parámetro** | `2` (fija) | `ln(n)` (crece con la muestra) |
| **Tendencia** | Favorece modelos algo más complejos | Más conservador: favorece modelos más simples |
| **Criterio** | Menor AIC = mejor | Menor BIC = mejor |

> Enlaza con [Estrategias de Selección de Variables](<Estrategias de Selección de Variables.md>) (Clase 17) y [Selección de Variables con Regularización](<Selección de Variables con Regularización.md>) (Clase 19).
