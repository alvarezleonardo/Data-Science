# Cobertura de los entregables

> Contraste entre lo que esta hecho y lo que piden las cuatro etapas del Desafio Profesional, mas la
> revision de si el enfoque elegido sigue respondiendo al enunciado del caso. Consignas en
> [../../../consignas/](../../../consignas/); enunciado del caso en
> [2.1 - Casos de Negocio - Subtes.md](<../2.1 - Casos de Negocio - Subtes.md>).

## 1. Resumen

| Etapa | Entregable | Estado |
|---|---|---|
| **1** | Documento de EDA con estadistica descriptiva, outliers y patrones | Parcial |
| **1** | Scripts de Python | Parcial |
| **1** | Visualizaciones | **Falta** |
| **1** | Documentacion del proceso | Hecho |
| **2** | Scripts de limpieza y transformacion en Python | Hecho |
| **2** | Consultas **SQL** | **Falta** |
| **2** | Dataset final transformado | Parcial |
| **2** | Documentacion del ETL | Hecho |
| **3** | Scripts de modelado | **Falta** |
| **3** | Evaluacion del modelo con metricas y justificacion | **Falta** |
| **3** | Documentacion del modelado | **Falta** |
| **4** | Script de la red neuronal | **Falta** |
| **4** | Comparacion de Machine Learning contra redes neuronales | **Falta** |
| **4** | **Video explicativo de 5 minutos** | **Falta** |
| **4** | Documentacion del modelado de redes | **Falta** |

Lo que esta solido es el trabajo de **datos y diagnostico**, que es la base de todo el resto y esta mas
avanzado de lo que se pide a esta altura. Lo que no existe todavia es **la mitad visible del trabajo**:
notebooks, graficos y modelos. Conviene decirlo asi, sin disimularlo: hay siete documentos y un script,
pero **ni un grafico ni un modelo**.

## 2. Cuatro huecos que no son "falta hacerlo" sino decisiones pendientes

### 2.1 El entorno no tiene pandas, matplotlib ni seaborn

Verificado: el `python3` disponible es el de Homebrew y no los tiene; el entorno `~/.venvs/dl-env` que
figura en las notas de sesiones anteriores **no existe en esta maquina**. El ETL se escribio con la
libreria estandar justamente para no depender de eso.

Pero las tres etapas restantes son inviables sin ese stack, y la consigna de la Etapa 1 evalua
explicitamente la "eficiencia en el uso de herramientas (Pandas, Matplotlib, Seaborn)". **Es el primer
bloqueante a resolver**, antes que cualquier otra cosa.

### 2.2 Falta SQL, y la Etapa 2 lo pide por nombre

El entregable dice "codigo en Python **y consultas SQL**". Todo el trabajo hasta ahora es sobre
archivos planos: no hay una sola consulta.

La salida mas simple es cargar `agregado-horario.csv` en **SQLite**, que viene incluido en Python y no
necesita instalar ni administrar nada, y escribir ahi las agregaciones del analisis. Como beneficio
adicional, las consultas por grupo de estaciones y franja horaria quedan mucho mas legibles en SQL que
en el `awk` con el que se calcularon los resultados preliminares, y pasan a ser parte del entregable en
lugar de comandos sueltos.

### 2.3 El analisis es explicativo, pero las etapas 3 y 4 exigen predecir

Este es el hueco conceptual, y el mas importante de los cuatro.

El [diseno actual](diseno-analisis-flujo.md) responde **que paso y por que**: compara perfiles, mide
asimetrias, contrasta hipotesis. Es un trabajo descriptivo y explicativo, y los
[resultados](resultados-preliminares.md) muestran que funciona. Pero la Etapa 3 pide modelos con
metricas de evaluacion y justificacion del modelo elegido, y la Etapa 4 pide una red neuronal
comparada contra ellos. **Ninguna de las dos se satisface con estadistica descriptiva**, por buena que
sea.

Falta definir **que se predice**. La articulacion que resuelve el problema sin abandonar la pregunta
original:

> **Entrenar los modelos con los datos de 2014 a 2019 y usarlos para proyectar el contrafactual: que
> flujo habria tenido cada estacion si la pandemia no hubiera ocurrido. La brecha entre esa proyeccion
> y lo efectivamente observado es el impacto, cuantificado en pasajeros.**

Con eso, el requisito predictivo deja de ser un agregado artificial y se convierte en la herramienta
que contesta el "en que porcentaje" de la pregunta original:

- **Etapa 3.** Variable objetivo: pasajeros diarios por estacion. Predictores: dia de la semana, mes,
  feriado, estacion del anio, tarifa real, grupo funcional, tendencia. Modelos a comparar: regresion
  lineal y regularizada como linea base, arbol y random forest, y gradient boosting. Metricas: MAE,
  RMSE y MAPE sobre un conjunto de validacion **temporalmente posterior** al de entrenamiento, nunca
  aleatorio, porque es una serie temporal. Mas el **clustering** de estaciones por perfil horario, que
  el enunciado del caso pide de forma explicita, comparando el agrupamiento pre y post pandemia: si
  los clusters se reordenan, es evidencia directa de cambio estructural.
- **Etapa 4.** LSTM o GRU sobre la misma serie, con la misma particion temporal y las mismas metricas,
  para que la comparacion con la Etapa 3 sea legitima. La hipotesis a contrastar es si la red captura
  mejor la estacionalidad multiple (semanal y anual) que los modelos tabulares.

### 2.4 Hay partes del enunciado del caso que el enfoque dejo afuera

El brief propone cuatro preguntas orientativas y permite reemplazarlas, pero dos de ellas se pueden
cubrir casi sin costo adicional y conviene no perderlas:

| Pregunta del brief | Estado | Costo de incorporarla |
|---|---|---|
| Grupos de estaciones con patrones de uso similares (clustering) | Cubierta por el plan de la Etapa 3 | Ya previsto |
| Estaciones con cambios segun dia y hora (series temporales) | **Ya respondida** en los resultados preliminares | Hecho |
| Pasajeros esperados en un dia tipico (modelos predictivos) | Queda cubierta por la variable objetivo propuesta | Ya previsto |
| **Influencia de la tarifa en la demanda (regresion)** | **No abordada** | Bajo: el dato ya esta descargado |

Sobre la tarifa: `registro-historico-del-precio-del-boleto.csv` tiene 305 registros desde 1994 y
**esta sin usar**. Incorporarla exige deflactar por inflacion para obtener la tarifa real, que es lo
unico no trivial, y a cambio aporta un predictor economico fuerte y cubre una pregunta explicita del
enunciado. Ademas sirve como **control de la conclusion principal**: si la tarifa real subio mucho en
el periodo, parte de la caida de pasajeros se explica por precio y no por trabajo remoto, y eso hay
que poder descartarlo.

## 3. El enfoque sigue respondiendo al enunciado

Dos chequeos de consistencia:

**Contra el enunciado del caso.** El brief pide analizar la red para "comprender la demanda de viajes,
los patrones de movilidad y proponer mejoras", y autoriza de forma expresa a plantear otras preguntas
("tu podras buscar responder otras que consideres relevantes"). El enfoque post-pandemia es una
especializacion legitima de "patrones de movilidad", y con el agregado de tarifa y clustering quedan
cubiertas tambien tres de las cuatro preguntas sugeridas. **No hay desvio respecto del enunciado.**

**Contra el encadenamiento de las etapas.** Las cuatro etapas suponen un unico caso trabajado de punta
a punta, donde cada una usa la salida de la anterior. La cadena esta bien formada:

```
Etapa 1  EDA sobre el agregado horario          -> entiende el dato y detecta los patrones
Etapa 2  normalizacion + comuna + tarifa        -> dataset analitico unico
Etapa 3  clustering + regresion con contrafactual-> cuantifica y agrupa
Etapa 4  LSTM contra los modelos de la etapa 3   -> compara enfoques
```

Lo que hay que corregir es el **orden de ejecucion**: se avanzo en el ETL de la Etapa 2 y en hallazgos
que corresponden a la Etapa 1 antes de haber producido el EDA formal de la Etapa 1. No es un problema
de fondo, porque el trabajo esta hecho y documentado, pero el entregable de la Etapa 1 hay que
escribirlo igual, y **con las visualizaciones que son la mitad de su nota**.

## 4. Riesgo que conviene registrar

Los numeros de [resultados-preliminares.md](resultados-preliminares.md) se calcularon con `awk` sobre
el agregado, no con pandas, y **ningun resultado esta todavia reproducido en un notebook**. Antes de
apoyar conclusiones en ellos hay que recalcularlos con el stack del curso, tanto porque es lo que se
evalua como porque un segundo calculo independiente es la forma mas simple de detectar un error en el
primero.

## 5. Orden de trabajo propuesto

1. **Crear el entorno** con pandas, matplotlib, seaborn, scikit-learn y jupyter. Bloquea todo lo demas.
2. **Notebook de la Etapa 1**: cargar el agregado, estadistica descriptiva, nulos, outliers, y las
   visualizaciones (serie temporal, perfiles horarios superpuestos, perfil semanal, ranking de
   estaciones, heatmap de correlacion). Recalcular ahi los resultados preliminares.
3. **Cerrar la Etapa 2**: cargar en SQLite y escribir las consultas; sumar la asignacion de comuna
   desde `bocas-de-subte.csv` y la tarifa real deflactada; dejar el dataset analitico final.
4. **Etapa 3**: clustering de estaciones y modelos de regresion con validacion temporal y
   contrafactual.
5. **Etapa 4**: LSTM o GRU, comparacion con la Etapa 3, y el video de cinco minutos.

El video es el unico entregable que no depende de nada tecnico y suele dejarse para el final: conviene
tener en cuenta que son cinco minutos sobre un trabajo que ya tiene siete documentos, de modo que el
guion requiere seleccionar, no resumir todo.
