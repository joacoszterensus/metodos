import numpy as np
from scipy.interpolate import lagrange, CubicSpline,griddata
import matplotlib.pyplot as plt

def fa(x):
    return (0.3 ** abs(x))*np.sin(4*x)-np.tanh(2*x)+2

def chebyshev(n, interval):
    a, b = interval
    k = np.arange(1, n+1)
    x = np.cos((2*k - 1) * np.pi / (2*n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x

originalPoints=100
x=np.linspace(-4,4,originalPoints)
y=fa(x)

interpolatedPoints=9
xInterpolated=np.linspace(-4,4,interpolatedPoints)
yInterpolated=fa(xInterpolated)

y_Lineal=griddata(xInterpolated,yInterpolated,x,method="linear")

poly = lagrange(xInterpolated, yInterpolated)
y_Lagrange = poly(x)

cs = CubicSpline(xInterpolated, yInterpolated)
y_CubicSpline = cs(x)


#chevysheb
xInterpolated_chevy= chebyshev(interpolatedPoints, [-4, 4])
yInterpolated_chevy=fa(xInterpolated_chevy)

Y_lineal_chevy=griddata(xInterpolated_chevy,yInterpolated_chevy,x,method="linear")

poly_chevy = lagrange(xInterpolated_chevy, yInterpolated_chevy)
yLagrange_chevy = poly_chevy(x)

for j in range(0,len(xInterpolated_chevy)-1):
        for k in range(j,len(xInterpolated_chevy)):
            if xInterpolated_chevy[j]>xInterpolated_chevy[k]:
                xInterpolated_chevy[j], xInterpolated_chevy[k] = xInterpolated_chevy[k], xInterpolated_chevy[j]
                yInterpolated_chevy[j], yInterpolated_chevy[k] = yInterpolated_chevy[k], yInterpolated_chevy[j]
                
cs_chevy = CubicSpline(xInterpolated_chevy, yInterpolated_chevy)
y_Cubics_chevypline_chevy = cs_chevy(x)


error_absoluto_Lineal_chevy = np.abs(Y_lineal_chevy - y)

error_absoluto_Lagrange_chevy = np.abs(yLagrange_chevy - y)

error_absoluto_Splines_chevy = np.abs(y_Cubics_chevypline_chevy - y)

error_absoluto_Lineal = np.abs(y_Lineal - y)

error_absoluto_Lagrange = np.abs(y_Lagrange - y)

error_absoluto_Splines = np.abs(y_CubicSpline - y)

# Crear la figura con dos subtramas
fig, axs = plt.subplots(2,3, figsize=(15, 8))

# Subtrama para las funciones
axs[0,0].plot(x, y, label="Original", color="green")
axs[0,0].plot(x, y_Lineal, label="Lineal", color="red")
axs[0,0].plot(x, Y_lineal_chevy, label="Lineal Chevysheb", color="blue")
axs[0,0].set_title('Comparación: f(x) VS Interpolaciones Lineales')
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('f(x)')
axs[0,0].grid(True)
axs[0,0].legend(loc="upper right")

# Subtrama para los errores
axs[1,0].plot(x, error_absoluto_Lineal, label="Error Absoluto ", color="red")
axs[1,0].plot(x, error_absoluto_Lineal_chevy, label="Error Absoluto chevysheb", color="blue")
axs[1,0].set_title('Errores absolutos de Interpolación Lineal')
axs[1,0].set_xlabel('x')
axs[1,0].set_ylabel('Error')
axs[1,0].grid(True)
axs[1,0].legend(loc="upper right")
axs[1,0].set_ylim(0,0.7)




# Subtrama para las funciones
axs[0,1].plot(x, y, label="Original", color="green")
axs[0,1].plot(x, y_CubicSpline, label="Splines ", color="red")
axs[0,1].plot(x, y_Cubics_chevypline_chevy, label="splines chevysheb", color="blue")
axs[0,1].set_title('Comparación: f(x) VS  splines')
axs[0,1].set_xlabel('x')
axs[0,1].set_ylabel('f(x)')
axs[0,1].grid(True)
axs[0,1].legend(loc="upper right")

# Subtrama para los errores
axs[1,1].plot(x, error_absoluto_Splines, label="Error Absoluto ", color="red")
axs[1,1].plot(x, error_absoluto_Splines_chevy, label="Error Absoluto chevysheb", color="blue")
axs[1,1].set_title('Errores absolutos de Interpolación splines')
axs[1,1].set_xlabel('x')
axs[1,1].set_ylabel('Error')
axs[1,1].grid(True)
axs[1,1].legend(loc="upper right")
axs[1,1].set_ylim(0,0.7)




# Subtrama para las funciones
axs[0,2].plot(x, y, label="Original", color="green")
axs[0,2].plot(x, y_Lagrange, label="Lagrange", color="red")
axs[0,2].plot(x, yLagrange_chevy, label="Lagrange chevysheb", color="blue")
axs[0,2].set_title('Comparación: f(x) VS  Lagrange')
axs[0,2].set_xlabel('x')
axs[0,2].set_ylabel('f(x)')
axs[0,2].grid(True)
axs[0,2].legend(loc="upper right")

# Subtrama para los errores
axs[1,2].plot(x, error_absoluto_Lagrange, label="Error Absoluto ", color="red")
axs[1,2].plot(x, error_absoluto_Lagrange_chevy, label="Error Absoluto chevysheb", color="blue")
axs[1,2].set_title('Errores absolutos de Interpolación Lagrange')
axs[1,2].set_xlabel('x')
axs[1,2].set_ylabel('Error')
axs[1,2].grid(True)
axs[1,2].legend()
axs[1,2].set_ylim(0,0.7)





plt.show()


