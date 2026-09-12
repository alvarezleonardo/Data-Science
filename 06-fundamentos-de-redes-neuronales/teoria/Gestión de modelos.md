# Gestión de modelos

> Conversión a Markdown de las slides del curso (unidad "Backpropagation y Gestión de modelos", bloque 02). El PDF original está en [`material/`](material/).

## 1. Persistencia de modelos

Luego de entrenar un modelo de red neuronal, conviene tener una forma de **conservarlo para su uso futuro sin tener que volver a entrenarlo**. En otras palabras, crear una **versión persistente** del modelo.

Para qué sirve, según la slide:

- **Evita reentrenar** cada vez que se quiera utilizar, ahorrando tiempo y recursos.
- Permite su **implementación en servidores**, donde puede procesar datos en tiempo real respondiendo a solicitudes de usuarios o aplicaciones.
- Permite **desplegar múltiples instancias** del modelo en distintos servidores o nodos, mejorando la capacidad de respuesta y reduciendo la latencia.
- Permite **mantener un historial** de sus evoluciones y **revertir a versiones anteriores** si una actualización introduce problemas.

Para exportar un modelo entrenado con scikit-learn se pueden usar las librerías **joblib** y **pickle**.

## 2. Joblib

- Es **recomendado** para guardar modelos entrenados de scikit-learn porque **maneja eficientemente objetos grandes**.
- Permite **comprimir** los archivos serializados para ahorrar espacio en disco.
- La serialización suele ser **más rápida y eficiente** que con pickle.

```python
import joblib                                              # 1. importar

joblib.dump(mlp, 'mi_modelo.joblib')                       # 2. guardar

model = joblib.load('mi_modelo.joblib')                    # 3. cargar

joblib.dump(mlp, 'mi_modelo_comprimido.joblib', compress=3) # 4. guardar comprimido
```

El nivel de `compress` va de 0 a 9.

> **Nota (medido con el modelo del notebook):** el archivo sin comprimir pesa 392.276 bytes y el comprimido, 375.378 — apenas un **4% menos**. Con un MLP chico como este la compresión casi no aporta y suma tiempo de guardado y de carga; rinde en modelos grandes. El notebook de la clase usa `compress=5` y la slide `compress=3`.

## 3. Pickle

Es la librería **estándar de Python** para serializar y deserializar objetos.

```python
import pickle                                   # 1. importar

with open('mi_modelo.pkl', 'wb') as archivo:    # 2. guardar ('wb' = escritura binaria)
    pickle.dump(mlp, archivo)

with open('mi_modelo.pkl', 'rb') as archivo:    # 3. cargar ('rb' = lectura binaria)
    model = pickle.load(archivo)
```

`pickle.dump(obj, file)` serializa el objeto `obj` y lo guarda en `file`; `pickle.load(file)` carga un objeto serializado desde `file`.

## 4. Dos advertencias que la slide no menciona

**Seguridad.** Deserializar un archivo pickle **ejecuta código arbitrario**: un `.pkl` malicioso puede correr lo que quiera al abrirlo. Nunca cargar un archivo de origen desconocido. Como `joblib` usa pickle por debajo, la advertencia vale para los dos. Cuando el modelo tiene que viajar entre sistemas o entre organizaciones, conviene un formato de intercambio como **ONNX** o **PMML**, que describen el modelo sin serializar objetos de Python.

**Compatibilidad de versiones.** Un modelo guardado con una versión de scikit-learn puede fallar al cargarse con otra, o —peor— cargar sin error y comportarse distinto. La recomendación oficial es **registrar junto al modelo** la versión de scikit-learn, de Python y de las dependencias con las que se entrenó. Para persistencia a largo plazo, guardar además los datos y el código de entrenamiento, de modo que el modelo se pueda reconstruir.

## 5. Qué se guarda y qué no

`joblib.dump(mlp, ...)` serializa el **estimador entrenado**: sus pesos (`coefs_`), sesgos (`intercepts_`) e hiperparámetros. **No** guarda el `StandardScaler` ni ninguna otra transformación previa.

Eso importa: si el modelo se entrenó con datos escalados y al cargarlo se le pasan datos crudos, las predicciones van a estar mal **sin ningún error visible**.

> **Nota (verificado, Iris con 2 atributos, `sklearn 1.9.0`):** el mismo modelo cargado desde disco da **96,67%** con los datos escalados y **36,67%** con los crudos. No hay excepción ni advertencia: procesa la entrada y devuelve predicciones normalmente. Guardado como `Pipeline`, recibe datos crudos y devuelve el 96,67% correcto. La forma correcta es guardar el `Pipeline` completo, no el estimador suelto:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), MLPClassifier(random_state=42)).fit(X_train, y_train)
joblib.dump(pipe, 'modelo_completo.joblib')   # guarda el escalado junto con la red
```

Del mismo modo, el **orden de los atributos** al predecir tiene que ser el mismo con el que se entrenó: el modelo recibe posiciones, no nombres de columna.
