<!-- Convertido automáticamente desde: Fundamentos de SVM.pdf -->

Fundamentos de SVM

01 Fundamentos de las Máquinas de Vectores de Soporte (SVM)

SVM es un algoritmo de aprendizaje supervisado utilizado para clasificación y
regresión. Desarrollado por Vladimir Vapnik y sus colegas en los años 90, tiene
como objetivo **encontrar el hiperplano que mejor separa las clases** en el
espacio de características.

Concepto básico y margen máximo

- **Hiperplano:** una línea (en 2D) o un plano (en 3D) que separa dos clases.
- **Margen:** la distancia entre el hiperplano y los puntos más cercanos de
  cualquier clase, conocidos como **vectores de soporte**.
- **Margen máximo:** SVM busca maximizar este margen para mejorar la
  generalización del modelo.

Función de costo y solución del problema

- **Función de costo:** minimizar la función de pérdida que incluye un término de
  regularización para maximizar el margen.

      L(w, b) = (1/2)·‖w‖² + C · Σᵢ max(0, 1 − yᵢ·(w·xᵢ + b))

- **Problema de optimización cuadrática:** SVM se formula como un problema de
  optimización cuadrática.
- **Multiplicadores de Lagrange:** se utilizan para convertir el problema de
  optimización en uno más manejable.
- **Resolución:** mediante técnicas de programación cuadrática.

02 Tipos de SVM

SVM Lineal:
- Descripción: utiliza un hiperplano lineal para separar las clases.
- Aplicaciones: adecuado para problemas donde las clases son linealmente
  separables.

SVM No Lineal:
- Descripción: utiliza funciones kernel para transformar los datos y encontrar un
  hiperplano no lineal en el espacio transformado.
- Kernels comunes:
  - Kernel Polinomial: adecuado para datos con relaciones polinomiales.
  - Kernel RBF (Función Base Radial): bueno para datos con relaciones complejas y
    no lineales.

Kernel Trick

    K(xᵢ, xⱼ) = exp(−γ · ‖xᵢ − xⱼ‖²)

- Concepto: una técnica que permite a las SVM realizar separaciones no lineales
  transformando los datos a un espacio de mayor dimensión **sin necesidad de
  calcular las coordenadas** de los datos en ese espacio.
- Función: buena para relaciones complejas y no lineales; `γ` (gamma) es un
  parámetro que define el alcance de la función.
