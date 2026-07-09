# Data Science

![Progreso](https://img.shields.io/badge/progreso-2%2F9%20m%C3%B3dulos-blue)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Código, proyectos y apuntes de mi **Certificación en Data Science** (Digital House).
Especialización en Machine Learning, Aprendizaje no supervisado, Deep Learning (CNNs, RNNs) y desarrollo end-to-end de Inteligencia Artificial.

> Espacio abierto para conectar y compartir conocimiento.

## Apuntes consolidados

Apuntes teóricos de todo el programa, sintetizados y corregidos, en un solo documento:
**[APUNTES-DATA-SCIENCE.md](APUNTES-DATA-SCIENCE.md)**

## Programa y avance

| # | Módulo | Unidades | Clases | Estado |
|---|--------|:--------:|:------:|--------|
| 01 | [Introducción a Machine Learning](01-introduccion-a-machine-learning/) | 7 | 21 | ✅ Aprobado (7/7) |
| 02 | [Desafío Profesional DS — Etapa 1](02-desafio-profesional-etapa-1/) | 1 | 3 | ✅ Aprobado (1/1) |
| 03 | [Modelado avanzado en Machine Learning](03-modelado-avanzado-en-machine-learning/) | 7 | 28 | 🔄 En curso (71% · 5/7) |
| 04 | [Aprendizaje no supervisado](04-aprendizaje-no-supervisado/) | 6 | 31 | ⬜ 0% |
| 05 | [Desafío Profesional DS — Etapa 2](05-desafio-profesional-etapa-2/) | 1 | 1 | ⬜ 0% |
| 06 | [Fundamentos de redes neuronales](06-fundamentos-de-redes-neuronales/) | 8 | 28 | ⬜ 0% |
| 07 | [Fundamentos de Deep Learning](07-fundamentos-de-deep-learning/) | 7 | 32 | ⬜ 0% |
| 08 | [Gestión de proyectos de IA](08-gestion-de-proyectos-de-ia/) | 7 | 18 | ⬜ 1% |
| 09 | [Desafío Profesional DS — Etapa 3](09-desafio-profesional-etapa-3/) | 1 | 1 | ⬜ 0% |
| | **Total** | **45** | **163** | |

Leyenda: ✅ Aprobado · 🔄 En curso · ⬜ Pendiente

## Estructura del repositorio

```
.
├── README.md                     ← este índice
├── APUNTES-DATA-SCIENCE.md       ← apuntes teóricos consolidados
├── requirements.txt              ← dependencias para reproducir el entorno
├── 01-introduccion-a-machine-learning/
│   ├── teoria/                   ← slides del curso (PDF + conversión .md)
│   ├── notebooks/                ← ejercicios, prácticas y TP
│   └── datasets/                 ← datasets del módulo
├── 02-desafio-profesional-etapa-1/
│   ├── introduccion/  consigna/  casos-de-negocio/
├── 03-modelado-avanzado-en-machine-learning/
│   ├── teoria/  notebooks/  datasets/
└── 04..09/                       ← módulos siguientes (placeholders)
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
