<!-- Convertido automáticamente desde: Introducción a los Kernels.pdf -->

Introducción a los Kernels

01 Kernels

Un kernel es una función que calcula un **producto escalar en un espacio de
características de mayor dimensión sin necesidad de transformar explícitamente los
datos**. Permite a los algoritmos lineales (como SVM) aprender relaciones no
lineales: transforma problemas no lineales en problemas lineales en un espacio de
características de mayor dimensión.

Funciones Kernel

- **Kernel Lineal:** `K(xᵢ, xⱼ) = xᵢ · xⱼ`.
  Aplicación: datos linealmente separables.
- **Kernel Polinomial:** `K(xᵢ, xⱼ) = (xᵢ · xⱼ + c)^d`.
  Aplicación: captura interacciones polinómicas entre características.
- **Kernel RBF (Función de Base Radial):** `K(xᵢ, xⱼ) = exp(−γ·‖xᵢ − xⱼ‖²)`.
  Aplicación: captura relaciones complejas en los datos.
- **Kernel Sigmoidal:** `K(xᵢ, xⱼ) = tanh(α·xᵢ · xⱼ + c)`.
  Aplicación: modela relaciones similares a las de redes neuronales.

02 Aplicaciones

- **Clasificación:** SVM no lineal para clasificar datos complejos. Ejemplo:
  reconocimiento de dígitos escritos a mano (MNIST).
- **Regresión:** SVM para regresión no lineal (SVR). Ejemplo: predicción de
  valores continuos con relaciones no lineales.
- **Reducción de dimensionalidad:** Análisis de Componentes Principales con Kernel
  (KPCA). Ejemplo: proyección de datos en espacios de menor dimensión manteniendo
  relaciones no lineales.

Ventajas:
- Flexibilidad para modelar relaciones no lineales.
- Mejor rendimiento en datos complejos.
- Amplia aplicabilidad en diferentes dominios.

Desventajas:
- La selección de parámetros puede ser complicada.
- Computacionalmente intensivo en grandes datasets.
- La interpretación de los resultados puede ser más difícil.
