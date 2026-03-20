import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline


#PUNTO 2-A

data = np.genfromtxt('mnyo_mediciones.csv')
x = data[:, 0]
y = data[:, 1]

t = np.arange(len(x))

ground_truth = np.genfromtxt('mnyo_ground_truth.csv')
x_gt = ground_truth[:, 0]
y_gt = ground_truth[:, 1]

poly_x = lagrange(t, x)
poly_y = lagrange(t, y)

cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)

t_interp = np.linspace(0, len(x) - 1, 1000)

x_interp_lagrange = poly_x(t_interp)
y_interp_lagrange = poly_y(t_interp)

x_interp_cs = cs_x(t_interp)
y_interp_cs = cs_y(t_interp)

plt.figure(figsize=(10, 6))
plt.plot(x_interp_lagrange, y_interp_lagrange, label='Interpolación con Lagrange', color='green')
plt.plot(x_interp_cs, y_interp_cs, label='Interpolación con Cubic Splines', color='red')
plt.plot(x_gt, y_gt, '--', label='Trayectoria real (ground truth)')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Comparación de trayectorias interpoladas')
plt.legend()
plt.grid(True)
plt.show()

#Punto 2-B

data1 = np.genfromtxt('mnyo_mediciones.csv')
data2 = np.genfromtxt('mnyo_mediciones2.csv')

# Extraer los tiempos y las posiciones para cada conjunto de datos
t1, x1 = data1[:, 0], data1[:, 1]
t2, x2 = data2[:, 0], data2[:, 1]

# Ordenar los datos por tiempo
t1_sorted = np.sort(t1)
t2_sorted = np.sort(t2)

# Crear los polinomios de Lagrange y Splines cúbicos para cada conjunto de datos
poly_x1 = lagrange(t1_sorted, x1)
spline_x1 = CubicSpline(t1_sorted, x1)

poly_x2 = lagrange(t2_sorted, x2)
spline_x2 = CubicSpline(t2_sorted, x2)

# Definir los puntos de tiempo para la interpolación
t_interp1 = np.linspace(t1_sorted[0], t1_sorted[-1], 1000)
t_interp2 = np.linspace(t2_sorted[0], t2_sorted[-1], 1000)

# Evaluar los polinomios en los nuevos puntos de tiempo
x_interp1 = poly_x1(t_interp1)
x_spline1 = spline_x1(t_interp1)

x_interp2 = poly_x2(t_interp2)
x_spline2 = spline_x2(t_interp2)

# Graficar las interpolaciones de Lagrange y Splines cúbicos de las dos trayectorias
plt.figure(figsize=(12, 6))
plt.plot(t_interp1, x_interp1, label='Interpolación Lagrange Trayectoria A', color='blue')
plt.plot(t_interp1, x_spline1, label='Interpolación Spline Cubico Trayectoria A', color='green')
plt.plot(t_interp2, x_interp2, label='Interpolación Lagrange Trayectoria B', color='red')
plt.plot(t_interp2, x_spline2, label='Interpolación Spline Cubico Trayectoria B', color='orange')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Interpolaciones de Lagrange y Splines Cúbicos de las dos trayectorias')
plt.legend()
plt.grid(True)
plt.show()

def biseccion(funcionA, funcionB, x1, x2):
    epsilon = 1*np.e**(-12)
    iteraciones = 0
    while abs(x1 - x2) > epsilon and iteraciones!=1000:
        iteraciones+=1
        x3 = (x1 + x2) / 2
        if (funcionA(x3) - funcionB(x3)) * (funcionA(x1) - funcionB(x1)) < 0:
            x2 = x3
        else:
            x1 = x3
    return [x1, funcionA(x1)], iteraciones

def puntoFijo(funcionA, funcionB, xAnt):
    #si evaluas entre g(1) y g(2) tiene que caer entre 1 y 2 ambos
    #g prima de 1 y 2 < 1
    epsilon = 1e-12
    iteraciones = 0
    xSig = xAnt + 1
    while abs(xSig - xAnt) > epsilon and iteraciones != 1000:
        xAnt = xSig
        xSig = xAnt - (funcionA(xAnt) - funcionB(xAnt))
        iteraciones += 1
    return [xSig, funcionA(xSig)], iteraciones

def newton(funcionA, funcionB, xAnt):
    epsilon = 1e-12
    xSig = xAnt + 1
    iteraciones = 0
    while (abs(xSig - xAnt) > epsilon) and (abs(funcionA(xSig) - funcionB(xSig)) > epsilon) and iteraciones<1000:
        xAnt = xSig
        derivadaA = (funcionA(xAnt + epsilon) - funcionA(xAnt - epsilon)) / (2 * epsilon)
        derivadaB = (funcionB(xAnt + epsilon) - funcionB(xAnt - epsilon)) / (2 * epsilon)
        if derivadaA == 0 or derivadaB == 0:
            break
        xSig = xAnt - (funcionA(xAnt) - funcionB(xAnt)) / (derivadaA - derivadaB)
        iteraciones += 1
    return [xSig, funcionA(xSig)], iteraciones


def trayectoriasAtravezadas(funcionA, funcionB):
    """La funcion A y B se interseca
    Splines: Entre 8.8 y 9.8, entre 13 y 14, Entre 15.5 y 17
    Lagrange: Entre 0 y 0.5, entre 1 y 1.5, entre 8 y 9, entre 13 y 14, entre 16.5 y 17.5"""
    x1 = 0
    while x1 < 25:
        x2 = x1 + 0.2
        if (funcionA(x2) - funcionB(x2)) * (funcionA(x1) - funcionB(x1)) < 0:
            print("Raiz")
            interseccion, iteraciones = biseccion(funcionA, funcionB, x1, x2)
            print("Las trayectorias se cruzan en x =", interseccion, "con bisección, en", iteraciones)
            if(x1<funcionA(x1)<x2 and x1<funcionA(x1)<x2):
                interseccion, iteraciones = puntoFijo(funcionA, funcionB, x2)
                print("Las trayectorias se cruzan en x =", interseccion, "con puntoFijo, en", iteraciones)
            else:
                print("El teorema no aplica")
            interseccion, iteraciones = newton(funcionA, funcionB, x2)
            print("Las trayectorias se cruzan en x =", interseccion, "con newton, en", iteraciones)
            print("\n")
        x1 = x2
    print("\n")

print("Lagrange")
trayectoriasAtravezadas(poly_x1, poly_x2) # Lagrange
print("Splines")
trayectoriasAtravezadas(spline_x1, spline_x2) # Splines
