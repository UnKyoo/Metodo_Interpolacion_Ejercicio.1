# Código que implementa el esquema numérico  
# de interpolación para determinar la raíz de
# una ecuación

#            Autor:
# María Alexandra Guardia Canche
# Versión 1.0 : 17/02/2025

import numpy as np  # Se importa la librería NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Se importa Matplotlib para graficar

# Función actualizada
def f(x):
    return x**3 - 6*x**2 + 11*x - 6  # Se define la función cuya raíz queremos encontrar

# Interpolación de Lagrange
def lagrange_interpolation(x, x_puntos, y_puntos):
    P_Interpolacion = len(x_puntos)  # Se obtiene el número de puntos de interpolación
    Resultado = 0  # Inicializa el polinomio interpolante en 0
    for Iteracion_i in range(P_Interpolacion):  # Itera sobre los puntos de interpolación
        termino_Lag = y_puntos[Iteracion_i]  # Toma el valor correspondiente en y
        for Iteracion_j in range(P_Interpolacion):  # Segunda iteración para construir los términos de Lagrange
            if Iteracion_i != Iteracion_j:  # Se omite el término donde i == j
                termino_Lag *= (x - x_puntos[Iteracion_j]) / (x_puntos[Iteracion_i] - x_puntos[Iteracion_j])
        Resultado += termino_Lag  # Suma el término calculado al polinomio interpolante
    return Resultado  # Devuelve el valor interpolado

# Método de Bisección para encontrar la raíz de una función
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:  # Verifica si hay cambio de signo en el intervalo
        raise ValueError("El intervalo no contiene una raíz")  # Si no hay, se lanza un error

    for _ in range(max_iter):  # Se itera hasta alcanzar el número máximo de iteraciones
        c = (a + b) / 2  # Se calcula el punto medio del intervalo
        if abs(func(c)) < tol or (b - a) / 2 < tol:  # Se verifica si la solución es suficientemente precisa
            return c  # Retorna la raíz aproximada
        if func(a) * func(c) < 0:  # Si hay cambio de signo en [a, c], se ajusta el intervalo
            b = c
        else:  # Si no, la raíz está en [c, b]
            a = c
    return (a + b) / 2  # Retorna la mejor estimación después de iterar

# Selección de tres puntos de interpolación en el intervalo [1,3]
x0 = 1.0  # Primer punto
x1 = 2.0  # Segundo punto
x2 = 3.0  # Tercer punto
x_points = np.array([x0, x1, x2])  # Se almacenan los puntos en un array
y_points = f(x_points)  # Se evalúa la función en los puntos seleccionados

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x2, 100)  # Se generan 100 puntos entre x0 y x2 para graficar
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]  # Se evalúa la interpolación en cada punto

# Encontrar raíz del polinomio interpolante usando bisección
root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)

# Calcular errores
errores_absolutos = np.abs(y_interp - f(x_vals))  # Se calcula el error absoluto
errores_relativos = errores_absolutos / np.where(np.abs(f(x_vals)) == 0, 1, np.abs(f(x_vals)))  # Se calcula el error relativo evitando división por cero
errores_cuadraticos = errores_absolutos**2  # Se calcula el error cuadrático

# Encabezado de la tabla de errores
print(f"{'Iteración':<10}|{'x':<12}|{'Error absoluto':<18}|{'Error relativo':<18}|{'Error cuadrático'}")
print("-" * 80)

# Iterar sobre los valores calculados para imprimirlos
for i, (x_val, error_abs, error_rel, error_cuad) in enumerate(zip(x_vals, errores_absolutos, errores_relativos, errores_cuadraticos)):
    print(f"{i+1:<10}|{x_val:<12.6f}|{error_abs:<18.6e}|{error_rel:<18.6e}|{error_cuad:.6e}")

# Se genera la gráfica de los errores
fig, ax = plt.subplots(1, 2, figsize=(14, 5))  # Crea dos subgráficos en una figura de tamaño 14x5

# Subgráfica 1: Errores
ax[0].plot(x_vals, errores_absolutos, label="Error Absoluto", color='purple')  # Grafica el error absoluto
ax[0].plot(x_vals, errores_relativos, label="Error Relativo", color='orange')  # Grafica el error relativo
ax[0].plot(x_vals, errores_cuadraticos, label="Error Cuadrático", color='brown')  # Grafica el error cuadrático
ax[0].set_xlabel("x")  # Etiqueta del eje x
ax[0].set_ylabel("Errores")  # Etiqueta del eje y
ax[0].legend()  # Muestra la leyenda
ax[0].grid(True)  # Activa la cuadrícula

# Subgráfica 2: Función original e interpolación
ax[1].plot(x_vals, f(x_vals), label="f(x) = x³ - 6x² + 11x - 6", linestyle='dashed', color='blue')  # Se grafica la función original
ax[1].plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')  # Se grafica la interpolación
ax[1].axhline(0, color='black', linewidth=0.5, linestyle='--')  # Línea horizontal en y = 0
ax[1].axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")  # Se marca la raíz encontrada
ax[1].scatter(x_points, y_points, color='black', label="Puntos de interpolación")  # Se marcan los puntos de interpolación
ax[1].set_xlabel("x")  # Etiqueta del eje x
ax[1].set_ylabel("f(x)")  # Etiqueta del eje y
ax[1].set_title("Interpolación y búsqueda de raíces")  # Título del gráfico
ax[1].legend()  # Muestra la leyenda
ax[1].grid(True)  # Activa la cuadrícula

# Guardar la gráfica en un archivo
plt.savefig("interpolacion_raices.png")  # Guarda la imagen de la gráfica
plt.show()  # Muestra la gráfica en pantalla

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")  # Muestra la raíz en la terminal
