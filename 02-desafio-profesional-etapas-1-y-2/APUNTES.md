# Apuntes — Módulo 02: Desafío Profesional · Etapa 1

> Guía de la primera etapa del proyecto integrador. Repaso global en [`../manual/`](../manual/README.md).

## Objetivo de la Etapa 1

Elegir un **caso de negocio** y aplicar el inicio del pipeline de Data Science: comprender el problema, explorar visualmente los datos (EDA) y dejarlos limpios y transformados para el modelado posterior.

## Clases

1. **Casos de Negocio — Introducción DS:** marco del desafío y elección del caso.
2. **Exploración Visual de los Datos (EDA):** entender distribución, relaciones, outliers y faltantes con estadística descriptiva y gráficos.
3. **Limpieza y Transformación de Datos:** imputación de faltantes, encoding de categóricas, escalado/normalización, tratamiento de outliers y feature engineering inicial.

## Casos de negocio disponibles

Ver el detalle en [`../desafio-profesional/casos-de-negocio/`](../desafio-profesional/casos-de-negocio/):

| Caso | Dominio | Problema sugerido |
|------|---------|-------------------|
| Subtes | Movilidad urbana (CABA) | Series temporales / demanda |
| Airbnb | Precios de alojamiento | Regresión de precio |
| Cambio Climático | Ambiental | Regresión / tendencias |
| Diabetes | Salud | Clasificación binaria |

## Checklist de EDA y limpieza

- [ ] Dimensiones del dataset, tipos de variable y memoria.
- [ ] Estadística descriptiva (`describe`) y distribución de la variable target.
- [ ] Valores faltantes: porcentaje por columna y estrategia (eliminar/imputar).
- [ ] Outliers: detección (IQR / z-score) y decisión.
- [ ] Correlaciones entre features y con el target.
- [ ] Encoding de categóricas y escalado de numéricas.
- [ ] Dataset final reproducible y documentado.

> Esta etapa es la base de la [Etapa 3](../05-desafio-profesional-etapa-3/) (modelado ML) y la [Etapa 4](../09-desafio-profesional-etapa-4/) (redes neuronales). Guía completa del desafío: [`../desafio-profesional/README.md`](../desafio-profesional/README.md).
