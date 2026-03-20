import numpy as np
import matplotlib.pyplot as plt

def diferencial_exponencial(n,t):
    return reproduccion*n

def diferencial_logaritmica(n,t):
    return reproduccion * n * (maximaPoblacion - n) / maximaPoblacion

def solucion_exponencial(t,n0):
    return n0*np.exp(t*reproduccion)

def solucion_Logistica(t,n0):
    return maximaPoblacion/(((maximaPoblacion-n0)/n0)*np.exp(-reproduccion*t)+1)

def euler(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t < tf:
        n = n + h*f(n, t)
        t += h
        evolucion.append(n)
    return evolucion

def eulerMejorado(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t < tf:
        n = n + h/2*(f(n, t) + f(n + h*f(t, n), t + h))
        t += h
        evolucion.append(n)
    return evolucion

def rungeKutta4(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t < tf:
        k1 = h*f(n, t)
        k2 = h*f(n + 1/2*k1, t + h/2)
        k3 = h*f(n + 1/2*k2, t + h/2)
        k4 = h*f(n + k3, t + h)
        n = n + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        evolucion.append(n)
    return evolucion


# Generamos datos para los gráficos
tiempo = np.linspace(0, 100, 100)  
poblacionInicial = 1
reproduccion = 0.1
maximaPoblacion = 2000





# POBLACION EN FUNCION DEL TIEMPO





fig, axs = plt.subplots(2, 1, figsize=(8, 6))  

# Primer subgráfico: Función exponencial
axs[0].set_title('Función exponencial')
poblacion_exponencial = solucion_exponencial(tiempo, poblacionInicial)
axs[0].plot(tiempo, poblacion_exponencial)

# Segundo subgráfico: Función logarítmica
axs[1].set_title('Función logarítmica')
poblacion_logaritmica = solucion_Logistica(tiempo, poblacionInicial)
axs[1].plot(tiempo, poblacion_logaritmica)

# Título general de la figura
fig.suptitle('Población en función del Tiempo', fontsize=16)

# Mostramos la figura
plt.tight_layout()
plt.show()





# VARIACION DE LA POBLACION EN FUNCION DE LA POBLACION





fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 filas, 1 columna

# Primer subgráfico: Función exponencial
axs[0].set_title('Función exponencial')
variacion_poblacion_exponencial = diferencial_exponencial( poblacion_exponencial,0)
axs[0].plot(poblacion_exponencial, variacion_poblacion_exponencial)

# Segundo subgráfico: Función logarítmica
axs[1].set_title('Función logarítmica')
variacion_poblacion_logaritmica = diferencial_logaritmica( poblacion_logaritmica,0)
axs[1].plot(poblacion_logaritmica, variacion_poblacion_logaritmica)

# Título general de la figura
fig.suptitle('Variación de la población en función de la población', fontsize=16)

# Mostramos la figura
plt.tight_layout()
plt.show()



# APROXIMACIONES




fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 filas, 1 columna


# Primer subgráfico: Función exponencial
axs[0].set_title('Función exponencial')
solucionRealExponencial=solucion_exponencial(tiempo,poblacionInicial,)
solucionEulerExponencial=euler(diferencial_exponencial,poblacionInicial,1,0,99)
solucionEulerMejoradoExponencial=eulerMejorado(diferencial_exponencial,poblacionInicial,1,0,99)
solucionRK4Exponencial=rungeKutta4(diferencial_exponencial,poblacionInicial,1,0,99)

axs[0].plot(tiempo, solucionRealExponencial, label='Funcion Original')  
axs[0].plot(tiempo, solucionEulerExponencial, label='Aproximacion  Euler') 
axs[0].plot(tiempo, solucionEulerMejoradoExponencial, label='Aproximacion Euler mejorado') 
axs[0].plot(tiempo, solucionRK4Exponencial, label='Aproximacion Runge Kutta 4')  
axs[0].set_title('Aproximaciones VS Original Exponencial')  # Título del primer subgráfico
axs[0].legend()  # Mostramos la leyenda en el primer subgráfico


# Segundo subgráfico: Función logarítmica
axs[1].set_title('Función logarítmica')
solucionRealLogistica=solucion_Logistica(tiempo,poblacionInicial)
solucionEulerLogistica=euler(diferencial_logaritmica,poblacionInicial,1,0,99)
solucionEulerMejoradoLogistica=eulerMejorado(diferencial_logaritmica,poblacionInicial,1,0,99)
solucionRK4Logistica=rungeKutta4(diferencial_logaritmica,poblacionInicial,1,0,99)

axs[1].plot(tiempo, solucionRealLogistica, label='Funcion Original')  
axs[1].plot(tiempo, solucionEulerLogistica, label='Aproximacion  Euler') 
axs[1].plot(tiempo, solucionEulerMejoradoLogistica, label='Aproximacion Euler mejorado') 
axs[1].plot(tiempo, solucionRK4Logistica, label='Aproximacion Runge Kutta 4')  
axs[1].set_title('Aproximaciones VS Original Logistica')  # Título del primer subgráfico
axs[1].legend()  # Mostramos la leyenda en el primer subgráfico

# Título general de la figura
fig.suptitle('Variación de la población en función de la población', fontsize=16)

# Mostramos la figura
plt.tight_layout()
plt.show()





#ERRORES

fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 filas, 1 columna


# Primer subgráfico: Función exponencial
solucionRealExponencial=solucion_exponencial(tiempo,poblacionInicial,)
solucionEulerExponencial=euler(diferencial_exponencial,poblacionInicial,1,0,99)
solucionEulerMejoradoExponencial=eulerMejorado(diferencial_exponencial,poblacionInicial,1,0,99)
solucionRK4Exponencial=rungeKutta4(diferencial_exponencial,poblacionInicial,1,0,99)

errorAbsolutoEuler= np.abs(solucionEulerExponencial - solucionRealExponencial)
errorAbsolutoEulerMejorado= np.abs(solucionEulerMejoradoExponencial - solucionRealExponencial)
errorAbsolutoRK4= np.abs(solucionRK4Exponencial - solucionRealExponencial)
axs[0].set_title('Función exponencial')
axs[0].plot(tiempo, errorAbsolutoEuler, label='Error absoluto Euler') 
axs[0].plot(tiempo, errorAbsolutoEulerMejorado, label='Error absoluto Euler mejorado') 
axs[0].plot(tiempo, errorAbsolutoRK4, label='Error absoluto Runge Kutta 4')  
axs[0].set_title('Error absoluto Exponencial')  # Título del primer subgráfico
axs[0].legend()  # Mostramos la leyenda en el primer subgráfico


# Segundo subgráfico: Función logarítmica
solucionRealLogistica=solucion_Logistica(tiempo,poblacionInicial)
solucionEulerLogistica=euler(diferencial_logaritmica,poblacionInicial,1,0,99)
solucionEulerMejoradoLogistica=eulerMejorado(diferencial_logaritmica,poblacionInicial,1,0,99)
solucionRK4Logistica=rungeKutta4(diferencial_logaritmica,poblacionInicial,1,0,99)

errorAbsolutoEuler= np.abs(solucionEulerLogistica - solucionRealLogistica)
errorAbsolutoEulerMejorado= np.abs(solucionEulerMejoradoLogistica - solucionRealLogistica)
errorAbsolutoRK4= np.abs(solucionRK4Logistica - solucionRealLogistica)

axs[1].plot(tiempo, errorAbsolutoEuler, label='Error Absoluto Euler') 
axs[1].plot(tiempo, errorAbsolutoEulerMejorado, label=' Error Absoluto Euler mejorado') 
axs[1].plot(tiempo, errorAbsolutoRK4, label='Error Absoluto Runge Kutta 4')  
axs[1].set_title('Error Absoluto Logistico')  # Título del primer subgráfico
axs[1].legend()  # Mostramos la leyenda en el primer subgráfico

# Título general de la figura
fig.suptitle('Error Absoluto De Las Aproximaciones', fontsize=16)

# Mostramos la figura
plt.tight_layout()
plt.show()

