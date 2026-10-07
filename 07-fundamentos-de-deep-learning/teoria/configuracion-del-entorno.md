# Configuración del entorno

> Clase 2 del programa. Las instrucciones del curso son para **Windows con VS Code**; se conservan completas en la sección 3. Las secciones 1 y 2 documentan la instalación real en **macOS con Apple Silicon**, que es donde se trabaja este repositorio.

## 1. La instalación de esta máquina

**Sistema:** macOS 26.6, arquitectura **arm64** (Apple Silicon) · Python 3.12.13

### Por qué un entorno dedicado

Instalar TensorFlow sobre el entorno base de Anaconda **falla**:

```
error: uninstall-no-record-file
Cannot uninstall numpy None
The package's contents are unknown: no RECORD file was found for numpy.
```

La causa: `numpy` fue instalado por **conda**, no por pip, así que no tiene archivo `RECORD` y pip no puede desinstalarlo para resolver sus dependencias. Forzarlo (`--ignore-installed`) deja dos numpy pisados y puede romper el entorno donde ya funcionan los notebooks de los módulos anteriores.

La salida limpia es un **entorno virtual dedicado**, que además es la práctica habitual para deep learning: las versiones de TensorFlow y PyTorch son exigentes y conviene aislarlas.

### Comandos usados

```bash
# 1. Crear el entorno
/opt/anaconda3/bin/python3 -m venv ~/.venvs/dl-env

# 2. Instalar todo
~/.venvs/dl-env/bin/pip install --upgrade pip
~/.venvs/dl-env/bin/pip install numpy pandas matplotlib seaborn scikit-learn \
    tensorflow torch torchvision torchaudio torchsummary ipykernel

# 3. Registrar el kernel para Jupyter
~/.venvs/dl-env/bin/python -m ipykernel install --user \
    --name dl-env --display-name "Python 3.12 (dl-env)"
```

### Resultado verificado

| Librería | Versión |
|---|---|
| Python | 3.12.13 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.0 |
| **tensorflow** | **2.21.0** |
| keras | 3.15.1 |
| **torch** | **2.14.0** |
| torchvision | 0.29.0 |
| torchaudio | 2.11.0 |
| torchsummary | instalado |

**Aceleración por hardware:**

- **PyTorch: MPS disponible** (`torch.backends.mps.is_available() → True`). Usa la GPU del Mac.
- **TensorFlow: solo CPU.** Ver la advertencia de abajo.

### Advertencia: `tensorflow-metal` rompe TensorFlow

`tensorflow-metal` es el plugin que le daría acceso a la GPU a TensorFlow en Mac. **No instalarlo**: su última versión no acompaña a TF 2.21 y al importar tensorflow explota con

```
NotFoundError: dlopen(.../libmetal_plugin.dylib):
Library not loaded: @rpath/_pywrap_tensorflow_internal.so
```

No es que la GPU no funcione: **TensorFlow deja de importar por completo**. Se probó, se verificó el fallo y se desinstaló. Si ya está instalado:

```bash
~/.venvs/dl-env/bin/pip uninstall -y tensorflow-metal
```

TensorFlow en CPU anda bien para lo que pide el curso — un `matmul` de 1000×1000 tarda 0,03 s.

### Kernels de Jupyter disponibles

| Kernel | Intérprete | Para qué |
|---|---|---|
| `Python 3.12 (dl-env)` | `~/.venvs/dl-env` | **este módulo**: TensorFlow y PyTorch |
| `Python 3.11 (ml_env)` | `/opt/anaconda3` | módulos anteriores, scikit-learn |

Al crear un notebook de este módulo hay que elegir **dl-env**.

## 2. Notas de macOS que el curso no cubre

- **No hay que tocar el `PATH`.** Es un problema de Windows; en Mac lo resuelven el instalador, Homebrew o Anaconda. Para verificar: `python3 --version` y `which python3`. El comando es `python3`, no `python`.
- **`msvc-runtime` no existe en macOS.** El fix que da el curso para el error de matplotlib instala las bibliotecas de Visual C++, exclusivas de Windows.
- **PyTorch usa MPS, no CUDA.** Donde los ejemplos del curso digan `cuda`, acá va `mps`:

```python
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Forma portable, funciona en cualquier maquina
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu')
```

- **TensorFlow ya trae soporte arm64 nativo** desde la 2.16: alcanza con `pip install tensorflow`. Las guías que mandan instalar `tensorflow-macos` están obsoletas.
- **`torchsummary` no se actualiza desde 2018.** Se instala porque lo pide el curso; si falla con versiones nuevas de PyTorch, el reemplazo mantenido es `torchinfo`:

```python
from torchsummary import summary          # el del curso
summary(modelo, (3, 224, 224))

from torchinfo import summary             # equivalente mantenido
summary(modelo, input_size=(1, 3, 224, 224))
```

- **Instalar con la ruta completa del `pip`.** Usar `pip` a secas instala en el intérprete que esté primero en el `PATH`, que puede no ser el que levanta el kernel de Jupyter. Es la causa más común del "lo instalé pero el notebook no lo encuentra".

## 3. Instrucciones originales del curso (Windows + VS Code)

### Verificar que Python se instaló correctamente

```bash
python --version
```

### Agregar Python al PATH

Rutas que hay que agregar a la variable de entorno **`Path`** si Python no se instaló correctamente. Reemplazar `User` por el nombre de usuario del sistema y **colocar la barra invertida `\` al final de cada ruta**:

```
C:\Users\User\AppData\Local\Programs\Python\Python312\Scripts\
C:\Users\User\AppData\Local\Programs\Python\Python312\
```

### Librerías para VS Code

```bash
pip install numpy
pip install pandas
python -m pip install -U matplotlib
pip install seaborn
pip install -U scikit-learn
pip install tensorflow
pip3 install torch torchvision torchaudio
pip install torchsummary
```

### IMPORTANTE

En caso de que la librería **matplotlib** genere un error al momento de ejecutar el código, abrir una nueva terminal y ejecutar:

```bash
pip install msvc-runtime
```

Luego reiniciar VS Code y volver a ejecutar el código.
