import numpy as np
import matplotlib.pyplot as plt

#Ecuaciones diferenciales y sus soluciones
def diferencialExponencial(r, N):
    return r*N

def solucionExponencial(r, t, N0):
    return N0*np.exp(r*t)

def diferencialLogistica(r, N, K):
    return r*N*(1-N/K)

def solucionLogistica(r, t, K, N0):
    return K/(((K-N0)/N0)*np.exp(-r*t)+1)


#Metodos de resolucion de ecuaciones diferenciales
def euler(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t <= tf:
        n = n + h*f(n, t)
        t += h
        evolucion.append(n)
    return evolucion

def eulerMejorado(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t <= tf:
        n = n + h/2*(f(n, t) + f(n + h*f(t, n), t + h))
        t += h
        evolucion.append(n)
    return evolucion

def rungeKutta4(f, N0, h, t0, tf):
    n = N0
    t = t0
    evolucion = [n]
    while t <= tf:
        k1 = h*f(n, t)
        k2 = h*f(n + 1/2*k1, t + h/2)
        k3 = h*f(n + 1/2*k2, t + h/2)
        k4 = h*f(n + k3, t + h)
        n = n + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        evolucion.append(n)
    return evolucion

#N vs T
#Usaremos las capacidades N0=10, k=1000 para contrastar crecimiento
#Usaremos las capacidades N0=1000, k=1100 para contrastar decrecimiento, y probaremos que pasa con N0=2000 k=1000 (exceso) en crecimiento
#Posibilidades (r): Creciente 0.2, Decreciente -0.1, Altamente creciente 1
#Tiempo que modele bien cada funcion segun convenga

def solExp(t, h, tf, r, N0):
    solExp=[]
    while t<=tf:
        solExp.append(solucionExponencial(r, t, N0))
        t+=h
    return solExp

def solLog(t, h, tf, r, N0, K):
    solLog=[]
    while t<=tf:
        solLog.append(solucionLogistica(r, t, K, N0))
        t+=h
    return solLog

def poblacionVStiempo():
    """Grafica la población en función del tiempo para las funciones exponenciales y logísticas.
    Se grafican cuatro casos: creciente, decreciente, excedida y población límite decreciente.
    Creciente busca modelar una población creciente, asi como decreciente una decreciente.
    En el caso de excedida, buscamos modelar una población que excede el límite para logística.
    Población límite decreciente busca modelar una población que decrece iniciando en su población máxima.
    Cada caso se modelará con el tiempo que mejor lo represente."""
    t1=np.linspace(0, 40, 21)
    t2=np.linspace(0, 60, 31)

    ExpCreciente = solExp(t=0, h=2, tf=60, r=0.2, N0=10)
    ExpDecreciente = solExp(t=0, h=2, tf=60, r=-0.1, N0=1000)
    ExpExcedido = solExp(t=0, h=2, tf=40, r=0.2, N0=2000)
    ExpLimiteDecreciente = solExp(t=0, h=2, tf=40, r=-0.1, N0=1000)

    LogCreciente = solLog(t=0, h=2, tf=60, r=0.2, N0=10, K=1000)
    LogDecreciente = solLog(t=0, h=2, tf=60, r=-0.1, N0=1000, K=1100)
    LogExcedido = solLog(t=0, h=2, tf=40, r=0.2, N0=2000, K=1000)
    LogLimiteDecreciente = solLog(t=0, h=2, tf=40, r=-0.1, N0=1000, K=1000)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    # Grafica las soluciones

    axs[0, 0].plot(t2, ExpCreciente, color='blue', label='Función Exponencial')
    axs[0, 0].plot(t2, LogCreciente, color='red', label='Función Logística')
    axs[0, 0].plot(0, 10, 'ko', markersize=5, label='N0=10')
    axs[0, 0].axhline(y=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[0, 0].plot(0, 0, color='white', label='K=1000')
    axs[0, 0].plot(0, 0, color='white', label='r=0.2')
    axs[0, 0].set_title('Creciente', fontsize=12)
    axs[0, 0].set_ylim(0, 5000)
    axs[0, 0].set_xlabel('Tiempo')
    axs[0, 0].set_ylabel('Población')
    axs[0, 0].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=10', 'K=1000', 'r=0.2'])

    axs[0, 1].plot(t2, ExpDecreciente, color='blue', label='Función Exponencial')
    axs[0, 1].plot(t2, LogDecreciente, color='red', label='Función Logística')
    axs[0, 1].plot(0, 1000, 'ko', markersize=5, label='N0=1000')
    axs[0, 1].axhline(y=1100, color='black', linestyle='--', dashes=(5, 5), label='K=1100')
    axs[0, 1].plot(0, 0, color='white', label='r=-0.1')
    axs[0, 1].set_title('Decreciente', fontsize=12)
    axs[0, 1].set_ylim(0, 1200)
    axs[0, 1].set_xlabel('Tiempo')
    axs[0, 1].set_ylabel('Población')
    axs[0, 1].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=1000', 'K=1100', 'r=-0.1'])

    axs[1, 0].plot(t1, ExpExcedido, color='blue', label='Función Exponencial')
    axs[1, 0].plot(t1, LogExcedido, color='red', label='Función Logística')
    axs[1, 0].plot(0, 2000, 'ko', markersize=5, label='N0=2000')
    axs[1, 0].axhline(y=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[1, 0].plot(0, 0, color='white', label='r=0.2')
    axs[1, 0].set_title('Excedido', fontsize=12)
    axs[1, 0].set_ylim(0, 4000)
    axs[1, 0].set_xlabel('Tiempo')
    axs[1, 0].set_ylabel('Población')
    axs[1, 0].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=2000', 'K=1000', 'r=0.2'])

    axs[1, 1].plot(t1, ExpLimiteDecreciente, color='blue', label='Función Exponencial')
    axs[1, 1].plot(t1, LogLimiteDecreciente, color='red', label='Función Logística')
    axs[1, 1].plot(0, 1000, 'ko', markersize=5, label='N0=1000')
    axs[1, 1].axhline(y=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[1, 1].plot(0, 0, color='white', label='r=-0.1')
    axs[1, 1].set_title('Población Limite Decreciente', fontsize=12)
    axs[1, 1].set_ylim(0, 2000)
    axs[1, 1].set_xlabel('Tiempo')
    axs[1, 1].set_ylabel('Población')
    axs[1, 1].legend(title='Info', loc='upper left', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=1000', 'K=1000', 'r=-0.1'])

    fig.suptitle('Población vs tiempo', fontsize=12)
    plt.tight_layout()
    plt.show()


def variacionVSpoblacion():
    """Grafica la variación de la población en función de la población para las funciones exponenciales y logísticas.
    Se grafican cuatro casos: creciente, decreciente, excedida y población límite decreciente. Mismos a PoblacionvsTiempo.
    Creciente busca modelar una población creciente, asi como decreciente una decreciente.
    En el caso de excedida, buscamos modelar una población que excede el límite para logística.
    Población límite decreciente busca modelar una población que decrece iniciando en su población máxima.
    Cada caso se modelará con el tiempo que mejor lo represente."""
    t1=np.linspace(0, 40, 21)
    t2=np.linspace(0, 60, 31)

    ExpCreciente = solExp(t=0, h=2, tf=60, r=0.2, N0=10)
    ExpDecreciente = solExp(t=0, h=2, tf=60, r=-0.1, N0=1000)
    ExpExcedido = solExp(t=0, h=2, tf=40, r=0.2, N0=2000)
    ExpLimiteDecreciente = solExp(t=0, h=2, tf=40, r=-0.1, N0=1000)

    LogCreciente = solLog(t=0, h=2, tf=60, r=0.2, N0=10, K=1000)
    LogDecreciente = solLog(t=0, h=2, tf=60, r=-0.1, N0=1000, K=1100)
    LogExcedido = solLog(t=0, h=2, tf=40, r=0.2, N0=2000, K=1000)
    LogLimiteDecreciente = solLog(t=0, h=2, tf=40, r=-0.1, N0=1000, K=1000)
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    axs[0,0].set_title('Creciente', fontsize=12)
    variacion_exponencial_altamente_creciente = []
    for poblacion in ExpCreciente:
        variacion_exponencial_altamente_creciente.append(diferencialExponencial(1, poblacion))
    variacion_logistica_altamente_creciente = []
    for poblacion in LogCreciente:
        variacion_logistica_altamente_creciente.append(diferencialLogistica(1, poblacion, 1000))
    axs[0,0].plot(ExpCreciente, variacion_exponencial_altamente_creciente, color='blue', label='Función Exponencial')
    axs[0,0].plot(LogCreciente, variacion_logistica_altamente_creciente, color='red', label='Función Logística')
    axs[0,0].plot(10, variacion_exponencial_altamente_creciente[0], 'ko', markersize=5, label='N0=10')
    axs[0,0].axvline(x=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[0,0].plot(0, 0, color='white', label='r=0.2')
    axs[0,0].plot(10, variacion_logistica_altamente_creciente[0], 'ko', markersize=5)
    axs[0,0].set_xlabel('Población')
    axs[0,0].set_ylabel('Variación de la población')
    axs[0,0].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=10', 'K=1000', 'r=0.2'])
    axs[0,0].set_xlim(0, 1100)
    axs[0,0].set_ylim(0, 500)


    axs[0,1].set_title('Decreciente', fontsize=12)
    variacion_exponencial_decreciente = []
    for poblacion in ExpDecreciente:
        variacion_exponencial_decreciente.append(abs(diferencialExponencial(-0.1, poblacion)))
    variacion_logistica_decreciente = []
    for poblacion in LogDecreciente:
        variacion_logistica_decreciente.append(abs(diferencialLogistica(-0.1, poblacion, 1100)))
    axs[0,1].plot(ExpDecreciente, variacion_exponencial_decreciente, color='blue', label='Función Exponencial')
    axs[0,1].plot(LogDecreciente, variacion_logistica_decreciente, color='red', label='Función Logística')
    axs[0,1].plot(1000, variacion_exponencial_decreciente[0], 'ko', markersize=5, label='N0=1000')
    axs[0,1].axvline(x=1100, color='black', linestyle='--', dashes=(5, 5), label='K=1100')
    axs[0,1].plot(0, 0, color='white', label='r=-0.1')
    axs[0,1].plot(1000, variacion_logistica_decreciente[0], 'ko', markersize=5)
    axs[0,1].set_xlabel('Población')
    axs[0,1].set_ylabel('Variación de la población')
    axs[0,1].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=1000', 'K=1100', 'r=-0.1'])
    axs[0,1].set_xlim(0, 1100)
    axs[0,1].set_ylim(0, 200)

    axs[1,0].set_title('Excedida', fontsize=12)
    variacion_exponencial_excedida = []
    for poblacion in ExpExcedido:
        variacion_exponencial_excedida.append(abs(diferencialExponencial(0.2, poblacion)))
    variacion_logistica_excedida = []
    for poblacion in LogExcedido:
        variacion_logistica_excedida.append(abs(diferencialLogistica(0.2, poblacion, 1000)))
    axs[1,0].plot(ExpExcedido, variacion_exponencial_excedida, color='blue', label='Función Exponencial')
    axs[1,0].plot(LogExcedido, variacion_logistica_excedida, color='red', label='Función Logística')
    axs[1,0].plot(2000, variacion_exponencial_excedida[0], 'ko', markersize=5, label='N0=2000')
    axs[1,0].axvline(x=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[1,0].plot(0, 0, color='white', label='r=0.2')
    axs[1,0].plot(2000, variacion_logistica_excedida[0], 'ko', markersize=5)
    axs[1,0].set_xlabel('Población')
    axs[1,0].set_ylabel('Variación de la población')
    axs[1,0].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=2000', 'K=1000', 'r=0.2'])
    axs[1,0].set_ylim(bottom=0)
    axs[1,0].set_xlim(0, 4000)
    axs[1,0].set_ylim(-100, 1000)

    axs[1,1].set_title('Población Limite Decreciente', fontsize=12)
    variacion_exponencial_limite_decreciente = []
    for poblacion in ExpLimiteDecreciente:
        variacion_exponencial_limite_decreciente.append(abs(diferencialExponencial(-0.1, poblacion)))
    variacion_logistica_limite_decreciente = []
    for poblacion in LogLimiteDecreciente:
        variacion_logistica_limite_decreciente.append(abs(diferencialLogistica(-0.1, poblacion, 1000)))
    axs[1,1].plot(ExpLimiteDecreciente, variacion_exponencial_limite_decreciente, color='blue', label='Función Exponencial')
    axs[1,1].plot(LogLimiteDecreciente, variacion_logistica_limite_decreciente, color='red', label='Función Logística')
    axs[1,1].plot(1000, variacion_exponencial_limite_decreciente[0], 'ko', markersize=5, label='N0=1000')
    axs[1,1].axvline(x=1000, color='black', linestyle='--', dashes=(5, 5), label='K=1000')
    axs[1,1].plot(0, 0, color='white', label='r=-0.1')
    axs[1,1].plot(1000, variacion_logistica_limite_decreciente[0], 'ko', markersize=5)
    axs[1,1].set_xlabel('Población')
    axs[1,1].set_ylabel('Variación de la población')
    axs[1,1].legend(title='Info', loc='upper right', fontsize='small', labels=['Función Exponencial', 'Función Logística', 'N0=1000', 'K=1000', 'r=-0.1'])

    # Título general de la figura
    fig.suptitle('Variación de la población en función de la población', fontsize=12)

    # Mostramos la figura
    plt.tight_layout()
    plt.show()

def main():
    poblacionVStiempo()
    variacionVSpoblacion()

if __name__ == '__main__':
    main()