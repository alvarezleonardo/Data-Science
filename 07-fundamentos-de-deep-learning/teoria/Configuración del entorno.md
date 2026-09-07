# Configuración del entorno

> Clase 2 del programa. Transcripción de las instrucciones del curso, más el equivalente para macOS, que es el sistema donde se trabaja este repositorio.

## 1. Instrucciones del curso (Windows + VS Code)

### Verificar que Python se instaló correctamente

```bash
python --version
```

### Agregar Python al PATH

Si Python no quedó bien instalado, hay que agregar estas dos rutas a la variable de entorno **`Path`**. Reemplazar `User` por el nombre de usuario del sistema y **dejar la barra invertida `\` al final de cada ruta**:

```
C:\Users\User\AppData\Local\Programs\Python\Python312\Scripts\
C:\Users\User\AppData\Local\Programs\Python\Python312\
```

### Librerías

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

### Si matplotlib da error

Abrir una terminal nueva y ejecutar:

```bash
pip install msvc-runtime
```

Luego reiniciar VS Code y volver a ejecutar el código.

> `msvc-runtime` instala las bibliotecas de tiempo de ejecución de Visual C++, que en Windows hacen falta para las partes compiladas de matplotlib. **Es exclusivo de Windows**: en macOS y Linux ese paquete no existe ni se necesita.

## 2. Equivalente en macOS

En Mac no hay que tocar el `PATH` a mano: el instalador de Python, Homebrew o Anaconda lo resuelven solos. Para verificar:

```bash
python3 --version        # en macOS el comando es python3, no python
which python3            # muestra qué interprete se está usando
```

Las librerías se instalan igual, con `pip`, salvo dos diferencias:

```bash
pip3 install numpy pandas matplotlib seaborn scikit-learn
pip3 install tensorflow
pip3 install torch torchvision torchaudio
pip3 install torchsummary
```

### Apple Silicon (M1/M2/M3/M4)

En los Mac con chip ARM hay dos cosas propias:

- **TensorFlow**: desde la versión 2.16 el paquete `tensorflow` ya trae soporte nativo para arm64, así que alcanza con `pip3 install tensorflow`. Para aprovechar la GPU integrada se agrega `pip3 install tensorflow-metal`. (Las guías viejas que mandan instalar `tensorflow-macos` quedaron obsoletas.)
- **PyTorch**: se instala igual, y detecta la GPU a través del backend **MPS** (*Metal Performance Shaders*) en lugar de CUDA. En el código, donde los ejemplos del curso digan `cuda`, acá va `mps`:

```python
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

Una forma portable, que funciona en cualquier máquina:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu')
```

## 3. Estado del entorno de este repositorio

Verificado sobre el intérprete de Anaconda que usan los notebooks (`/opt/anaconda3/bin/python3`), en macOS 26.6 con arquitectura **arm64**:

| Librería | Estado | Versión |
|---|---|---|
| Python | OK | 3.12.13 |
| numpy | OK | 2.5.0 |
| pandas | OK | 3.0.4 |
| matplotlib | OK | 3.11.0 |
| seaborn | OK | 0.13.2 |
| scikit-learn | OK | 1.9.0 |
| **tensorflow** | **falta** | — |
| **torch / torchvision / torchaudio** | **falta** | — |
| **torchsummary** | **falta** | — |

Las cinco librerías del módulo anterior ya están; **faltan exactamente las de deep learning**. Para instalarlas en este entorno:

```bash
/opt/anaconda3/bin/pip install tensorflow tensorflow-metal
/opt/anaconda3/bin/pip install torch torchvision torchaudio
/opt/anaconda3/bin/pip install torchsummary
```

> Conviene usar la ruta completa del `pip` de Anaconda en lugar de `pip` a secas, para asegurarse de instalar en el mismo intérprete que levanta el kernel `ml_env` de Jupyter — que apunta a `/opt/anaconda3/bin/python`. Instalar en otro intérprete es la causa más común del "lo instalé pero el notebook no lo encuentra".

Para verificar después de instalar:

```python
import tensorflow as tf, torch
print('TensorFlow', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))
print('PyTorch', torch.__version__, '| MPS:', torch.backends.mps.is_available())
```

## 4. Nota sobre `torchsummary`

El curso pide `torchsummary`, un paquete que **no recibe actualizaciones desde 2018**. El reemplazo mantenido es **`torchinfo`**, con la misma idea pero mejor soporte:

```python
# torchsummary (el del curso)
from torchsummary import summary
summary(modelo, (3, 224, 224))

# torchinfo (equivalente mantenido)
from torchinfo import summary
summary(modelo, input_size=(1, 3, 224, 224))
```

Se instala el del curso para seguir las clases sin fricción; si algo falla con versiones nuevas de PyTorch, `torchinfo` es la salida.
