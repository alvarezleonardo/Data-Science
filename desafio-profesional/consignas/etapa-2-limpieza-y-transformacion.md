# Etapa II — Limpieza y Transformación de Datos

> Fuente: [`Etapa 2 - Limpieza y Transformación de Datos.pdf`](<../../02-desafio-profesional-etapas-1-y-2/consigna/Etapa 2 - Limpieza y Transformación de Datos.pdf>)

> El PDF titula esta diapositiva "Etapa I" (igual que la de la Etapa 1), lo que parece un error de numeración en el material original — el resto del documento y el índice se refieren consistentemente a la Etapa de Limpieza y Transformación de Datos.

## Objetivos

- **Calidad de los Datos:** el objetivo principal de esta etapa es asegurar que los datos sean de la más alta calidad posible, eliminando inconsistencias, manejando valores faltantes, y resolviendo cualquier problema de integridad referencial.
- **Preparación para el Análisis:** se espera que los estudiantes transformen los datos de manera que estén listos para un análisis más profundo y para la creación de visualizaciones avanzadas en la siguiente etapa.
- **Documentación y Reproducibilidad:** es crucial que cada paso del proceso de limpieza y transformación esté bien documentado para garantizar que el trabajo sea reproducible.

## Contenidos y Herramientas Clave

**SQL y Python**

- **SQL:** para la extracción de datos, creación de nuevas tablas, y realización de consultas que apoyen la limpieza de datos.
- **Pandas:** para la manipulación y transformación de datos, manejo de valores nulos, y creación de nuevas columnas derivadas.

**Técnicas de Limpieza de Datos:**

- Detección y manejo de valores faltantes.
- Identificación y tratamiento de outliers.
- Normalización y escalado de variables.

> **Tips**
> - **Revisión de Consistencia:** asegúrate de revisar la consistencia entre diferentes tablas o fuentes de datos.
> - **Automatización:** cuando sea posible, automatiza procesos de limpieza para aplicarlos a futuras versiones de los datos.

## Proceso de Limpieza de Datos

**Identificación de Valores Faltantes**
- Localiza y maneja valores nulos o faltantes en el dataset.

**Detección y Tratamiento de Outliers y Duplicados**
- Usa técnicas estadísticas y visuales para identificar outliers.
- Eliminación de duplicados o corrección de errores básicos en los datos.

**Normalización y Escalado de Datos**
- Ajusta las variables numéricas para que estén en la misma escala, si es necesario para el análisis posterior.

**Transformación de Variables**
- Creación de nuevas variables derivadas, agregación de datos, o cambios en el formato de los datos (por ejemplo, fechas).

> **Tips**
> - Utiliza `df.isnull().sum()` para identificar rápidamente las columnas con valores faltantes y `df.dropna()` o `df.fillna()` para tratarlos según el contexto.
> - Los boxplots y Z-scores son herramientas útiles para detectar outliers. Decide si eliminarlos, transformarlos o tratarlos de alguna otra manera.
> - Utiliza técnicas como Min-Max o Estandarización (Z-score) según el tipo de análisis que realizarás.
> - Usa `pd.to_datetime()` para manejar fechas y `groupby()` para agregaciones.

## Documentación del Proceso ETL

- **Extract:** detalla cómo se han extraído los datos de las fuentes originales, incluyendo las consultas SQL utilizadas.
- **Transform:** documenta cada paso del proceso de transformación, explicando por qué se tomaron ciertas decisiones (por ejemplo, eliminación de outliers, técnicas de normalización).
- **Load:** explica cómo se han almacenado los datos transformados, asegurando que están listos para ser utilizados en el análisis y visualización posteriores.

> **Tips**
> - **Clara Justificación:** justifica cada decisión de transformación de manera clara, explicando por qué se eligió una técnica específica para la limpieza o transformación de los datos.
> - **Reproducibilidad:** asegúrate de que todo el proceso esté bien documentado para que pueda ser reproducido por otros.

## Entregables Esperados

1. **Scripts de Limpieza y Transformación de Datos:** código en Python y consultas SQL que documentan cómo se han limpiado y transformado los datos.
   > Tip: asegúrate de que el código esté bien organizado, comentado, y que sea fácil de seguir.
2. **Dataset Final Transformado:** dataset limpio y transformado, listo para ser utilizado en la etapa de visualización.
   > Tip: verifica que todos los campos sean consistentes y que no haya errores o incoherencias en los datos finales.
3. **Documentación del Proceso de ETL:** documento que explique detalladamente cada paso del proceso de extracción, transformación y carga (ETL) de los datos.
   > Tip: mantén la documentación clara y concisa, pero lo suficientemente detallada para que cualquier persona pueda seguir el proceso.

## Evaluación

**Criterios de Evaluación**

- Calidad de los datos procesados, incluyendo la detección y manejo de outliers y valores faltantes.
- Eficiencia en la aplicación de transformaciones y técnicas de normalización.
- Documentación clara y completa del proceso ETL.
- Reproducibilidad del proceso de limpieza y transformación.

---

Con los datos ahora limpios y transformados, estarás preparado para la Etapa 3. Asegúrate de que todos los datos estén en las mejores condiciones posibles para facilitar el modelado.

> La limpieza y transformación de datos es una de las tareas más críticas en cualquier proyecto de análisis de datos. Hazlo bien ahora, y te facilitará enormemente las etapas posteriores del proyecto.
