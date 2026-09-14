# Data Science

![Progreso](https://img.shields.io/badge/completados-5%2F9%20m%C3%B3dulos-blue)
![En curso](https://img.shields.io/badge/en%20curso-m%C3%B3dulo%2007-orange)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Código, proyectos y apuntes de mi **Certificación en Data Science** (Digital House).
Especialización en Machine Learning, Aprendizaje no supervisado, Deep Learning (CNNs, RNNs) y desarrollo end-to-end de Inteligencia Artificial.

> Espacio abierto para conectar y compartir conocimiento.

## Apuntes consolidados

Apuntes teóricos de todo el programa, sintetizados y corregidos, en un solo documento:
**[manual/](manual/README.md)** — manual del programa, ordenado de lo más básico a lo más complejo: **58 capítulos en 9 partes**, con 36 diagramas y una referencia técnica de consulta. Partido en un archivo por parte; `APUNTES-DATA-SCIENCE.md` en la raíz queda como puntero al índice.

## Programa y avance

| # | Módulo | Unidades | Clases | Apuntes | Estado |
|---|--------|:--------:|:------:|:-------:|--------|
| 01 | [Introducción a Machine Learning](01-introduccion-a-machine-learning/) | 7 | 21 | [caps. 1-8, 10, 13-14](manual/01-fundamentos.md#parte-i--fundamentos) | ✅ Aprobado (7/7) |
| 02 | [Desafío Profesional DS — Etapas 1 y 2](02-desafio-profesional-etapas-1-y-2/) | 1 | 3 | — | ✅ Aprobado (1/1) |
| 03 | [Modelado avanzado en Machine Learning](03-modelado-avanzado-en-machine-learning/) | 7 | 28 | [caps. 9, 11-12](manual/03-modelos-lineales.md#parte-iii--modelos-lineales) | ✅ Aprobado (7/7) |
| 04 | [Aprendizaje no supervisado](04-aprendizaje-no-supervisado/) | 6 | 31 | [caps. 15-23](manual/06-aprendizaje-no-supervisado.md#parte-vi--aprendizaje-no-supervisado) | ✅ Aprobado (6/6) |
| 05 | [Desafío Profesional DS — Etapa 3](05-desafio-profesional-etapa-3/) | 1 | 1 | — | ⬜ Sin cursar · hay 1 PDF sin procesar |
| 06 | [Fundamentos de redes neuronales](06-fundamentos-de-redes-neuronales/) | 8 | 28 | [caps. 24-38](manual/07-redes-neuronales.md#parte-vii--redes-neuronales) | ✅ Completado (8/8) — teoría completa · 4 notebooks propios · evaluación final resuelta |
| 07 | [Fundamentos de Deep Learning](07-fundamentos-de-deep-learning/) | 7 | 33 | [caps. 39-45, 48-57](manual/08-deep-learning-con-frameworks.md#parte-ix--deep-learning-con-frameworks) | ✅ Aprobado (7/7) — teoría completa hasta autoencoders y GANs |
| 08 | [Gestión de proyectos de IA](08-gestion-de-proyectos-de-ia/) | 7 | 18 | — | ⬜ Sin cursar |
| 09 | [Desafío Profesional DS — Etapa 4](09-desafio-profesional-etapa-4/) | 1 | 1 | — | ⬜ Sin cursar |
| | **Total** | **45** | **164** | | **6 completados · 3 pendientes** |

Leyenda: ✅ Aprobado o completado · 🔄 En curso · ⬜ Pendiente

> El estado de cada módulo es el que declara su propio `README.md`; esta tabla lo resume.

Las 4 etapas del Desafío Profesional (módulos 02, 05 y 09) tienen una guía única de punta a punta en **[`desafio-profesional/`](desafio-profesional/)**: consignas, casos de negocio y checklist accionable por etapa.

## Estructura del repositorio

```
.
├── README.md                     ← este índice
├── APUNTES-DATA-SCIENCE.md       ← puntero al manual (ver manual/)
├── manual/                       ← manual del programa (58 capítulos, 9 partes)
├── requirements.txt              ← dependencias para reproducir el entorno
├── 01-introduccion-a-machine-learning/
│   ├── teoria/                   ← conversión a Markdown de las slides
│   │   └── material/             ← los PDF originales del curso
│   ├── notebooks/                ← ejercicios, prácticas y TP
│   └── datasets/                 ← datasets del módulo
├── 02-desafio-profesional-etapas-1-y-2/
│   ├── introduccion/  consigna/
├── 03-modelado-avanzado-en-machine-learning/
│   ├── teoria/  notebooks/  datasets/
├── 04-aprendizaje-no-supervisado/
│   ├── teoria/  notebooks/  datasets/
├── 05-desafio-profesional-etapa-3/   ← material recibido, sin procesar
├── 06-fundamentos-de-redes-neuronales/
│   ├── teoria/  notebooks/
│   └── practica/evaluacion-final/    ← enunciado + notebook resuelta
├── 07-fundamentos-de-deep-learning/
│   ├── teoria/  notebooks/  datasets/   ← módulo en curso
├── 08..09/                       ← módulos siguientes (placeholders)
└── desafio-profesional/          ← guía unificada de las 4 etapas del proyecto integrador
    ├── README.md  consignas/  casos-de-negocio/
```

Cada módulo con contenido tiene su propio `README.md` con el detalle de clases y materiales.

## Entorno

Los entornos virtuales (`.venv/`) y los datasets pesados (`*.zip`) **no se versionan**. Para reproducir el entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recomendado: **un único `.venv` en la raíz** del repo para todos los módulos (en vez de uno por ejercicio). La versión de Python sugerida está en [`.python-version`](.python-version).

Datasets pesados: ver [`scripts/descargar_datos.sh`](scripts/descargar_datos.sh).

## Convenciones

- Carpetas y archivos nuevos en `kebab-case`, sin acentos ni espacios.
- Cada slide del curso conserva su PDF original y una conversión `.md` al lado.
- Notebooks: un ejercicio = una carpeta con su notebook, dataset y `README.md`.
