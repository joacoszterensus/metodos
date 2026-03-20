import numpy as np
import matplotlib.pyplot as plt

def dN1_dt(N1, N2, r1, K1, alpha12):
    return r1 * N1 * (K1 - N1 - alpha12 * N2) / K1

def dN2_dt(N1, N2, r2, K2, alpha21):
    return r2 * N2 * (K2 - N2 - alpha21 * N1) / K2

def runge_kutta_4(f1, f2, N1_0, N2_0, r1, r2, K1, K2, alpha12, alpha21, h, num_steps):
    results_N1 = [N1_0]
    results_N2 = [N2_0]
    
    for _ in range(num_steps):
        N1 = results_N1[-1]
        N2 = results_N2[-1]
        
        k1_N1 = h * f1(N1, N2, r1, K1, alpha12)
        k1_N2 = h * f2(N1, N2, r2, K2, alpha21)
        
        k2_N1 = h * f1(N1 + 0.5 * k1_N1, N2 + 0.5 * k1_N2, r1, K1, alpha12)
        k2_N2 = h * f2(N1 + 0.5 * k1_N1, N2 + 0.5 * k1_N2, r2, K2, alpha21)
        
        k3_N1 = h * f1(N1 + 0.5 * k2_N1, N2 + 0.5 * k2_N2, r1, K1, alpha12)
        k3_N2 = h * f2(N1 + 0.5 * k2_N1, N2 + 0.5 * k2_N2, r2, K2, alpha21)
        
        k4_N1 = h * f1(N1 + k3_N1, N2 + k3_N2, r1, K1, alpha12)
        k4_N2 = h * f2(N1 + k3_N1, N2 + k3_N2, r2, K2, alpha21)
        
        N1_next = N1 + (1/6) * (k1_N1 + 2 * k2_N1 + 2 * k3_N1 + k4_N1)
        N2_next = N2 + (1/6) * (k1_N2 + 2 * k2_N2 + 2 * k3_N2 + k4_N2)
        
        if N1_next < 0:
            N1_next = 0

        if N2_next < 0:
            N2_next = 0

        results_N1.append(N1_next)
        results_N2.append(N2_next)
    
    return results_N1, results_N2

def plot_population(ax, tiempo, n1Poblacion, n2Poblacion, label,colour,title):
    ax.plot(tiempo, n1Poblacion, label=label,color=colour) 
    ax.plot(tiempo, n2Poblacion,color=colour) 
    ax.legend()
    ax.set_title(title)


def isoclinas_DN1(k1,alpha12,n2):
    return k1-alpha12*n2
def isoclinas_DN2(k2,alpha21,n1):
    return k2-alpha21*n1


def plot_isoclinas(ax, K1, K2, alpha12, alpha21,title):
    N1_values = np.linspace(0,  100, 100)
    N2_values = np.linspace(0,  100, 100)
    
    ax.plot(N1_values, isoclinas_DN2(K2, alpha21, N1_values), label="Isoclina especie 1", linestyle='--')
    ax.plot(N2_values, isoclinas_DN1(K1, alpha12, N2_values), label="Isoclina especie 2", linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.set_xlabel("Población 1")
    ax.set_ylabel("Población 2")


def casos():
    tiempo = np.linspace(0, 100, 400)
    r1 = 0.3
    r2 = 0.5 
    h = 1    
    num_steps = 399 


    fig, axs = plt.subplots(2,2, figsize=(18, 6))  # 2 fila, 2 columnas

    # Extinción de la especie 1
    # Gráfico de poblaciones
    #N1_0 > N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 25, 20, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs[0, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 > N2_0)", color="blue")
    axs[0, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 > N2_0)", color="red")
    #N1_0 < N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 20, 25, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs[0, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 < N2_0)", color="purple", linestyle='--')
    axs[0, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 < N2_0)", color="pink", linestyle='--')
    #N1_0 = N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 20, 20, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs[0, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 = N2_0)", color="green", linestyle=':')
    axs[0, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 = N2_0)", color="orange", linestyle=':')

    axs[0, 0].plot(0, 0, 'ko', markersize=0.0001, label='K1=50, K2=100, α12=1, α21=1')
    axs[0, 0].legend()
    axs[0, 0].set_title("Caso 1 - Extinción de la especie 1", fontsize=10)

    #Extinción de la especie 2
    #N1_0 > N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 100, 50, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs[0, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 > N2_0)", color="blue")
    axs[0, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 > N2_0)", color="red")
    #N1_0 < N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 100, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs[0, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 < N2_0)", color="purple", linestyle='--')
    axs[0, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 < N2_0)", color="pink", linestyle='--')
    #N1_0 = N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 50, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs[0, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 = N2_0)", color="green", linestyle=':')
    axs[0, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 = N2_0)", color="orange", linestyle=':')

    axs[0, 1].plot(0, 0, 'ko', markersize=0.0001, label='K1=100, K2=50, α12=1, α21=1')
    axs[0, 1].legend()
    axs[0, 1].set_title("Caso 2 - Extinción de la especie 2", fontsize=10)

    #Coexistencia inestable
    #N1_0 > N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 100, 50, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs[1, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 > N2_0)", color="blue")
    axs[1, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 > N2_0)", color="red")
    #N1_0 < N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 100, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs[1, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 < N2_0)", color="purple", linestyle='--')
    axs[1, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 < N2_0)", color="pink", linestyle='--')
    #N1_0 = N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 50, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs[1, 0].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 = N2_0)", color="green", linestyle=':')
    axs[1, 0].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 = N2_0)", color="orange", linestyle=':')
    axs[1, 0].plot(0, 0, 'ko', markersize=0.0001, label='K1=70, K2=100, α12=1, α21=2')
    axs[1, 0].legend()
    axs[1, 0].set_title("Caso 3 - Equilibrio inestable", fontsize=10)

    #Coexistencia estable
    #N1_0 > N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 100, 50, r1, r2, 75, 50, 1, 0.5, h, num_steps)
    axs[1, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 > N2_0)", color="blue")
    axs[1, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 > N2_0)", color="red")
    #N1_0 < N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 100, r1, r2, 75, 50, 1, 0.5, h, num_steps)
    axs[1, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 < N2_0)", color="purple", linestyle='--')
    axs[1, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 < N2_0)", color="pink", linestyle='--')
    #N1_0 = N2_0
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 50, r1, r2, 75, 50, 1, 0.5, h, num_steps)
    axs[1, 1].plot(tiempo, n1Poblacion, label="Población 1 (N1_0 = N2_0)", color="green", linestyle=':')
    axs[1, 1].plot(tiempo, n2Poblacion, label="Población 2 (N1_0 = N2_0)", color="orange", linestyle=':')
    axs[1, 1].plot(0, 0, 'ko', markersize=0.0001, label='K1=75, K2=50, α12=1, α21=0.5')
    axs[1, 1].legend()
    axs[1, 1].set_title("Caso 4 - Coexistencia estable", fontsize=10)

    fig.suptitle('Tiempo VS Poblacion', fontsize=12)


    plt.tight_layout()
    plt.show()

def isoclinas_puntos():
    r1 = 0.3
    r2 = 0.5 
    h = 1    
    num_steps = 399 

    fig2, axs2 = plt.subplots(2,2, figsize=(18, 6))  # 2 fila, 2 columnas
    fig2.suptitle('Curvas Isoclinas Cero y Campo Vectorial', fontsize=12)

    plot_isoclinas(axs2[0,0], 50, 100, 1, 1,"Caso 1 - Extinción especie 1")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 10, 10, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs2[0,0].plot(n1Poblacion,n2Poblacion)
    axs2[0,0].scatter(10,10,label="Condiciones iniciales",color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 40, 20, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs2[0,0].plot(n1Poblacion,n2Poblacion)
    axs2[0,0].scatter(40,20,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 70, 80, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs2[0,0].plot(n1Poblacion,n2Poblacion)
    axs2[0,0].scatter(70,80,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 40, 50, r1, r2, 50, 100, 1, 1, h, num_steps)
    axs2[0,0].plot(n1Poblacion,n2Poblacion)
    axs2[0,0].scatter(40,50,color="black")
    axs2[0,0].plot(0, 0, 'ko', markersize=0.0001, label='K1=50, K2=100, α12=1, α21=1')

    #Puntos
    estabilidadPuntos(K1=50, K2=100, alpha12=1, alpha21=1, N1=0, N2=100, caso = "Caso 1 - Extinción especie 1")
    axs2[0,0].plot(0, 100, 'ko', markersize=7, label='Puntos de equilibrio estable', color="#FFD700")
    estabilidadPuntos(K1=50, K2=100, alpha12=1, alpha21=1, N1=50, N2=0, caso = "Caso 1 - Extinción especie 2")
    axs2[0,0].plot(50, 0, 'ko', markersize=7, label='Puntos de equilibrio inestable', color="grey")

    plot_isoclinas(axs2[0,1],100, 50, 1, 1,"Caso 2 - Extinción especie 2")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 20, 20, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs2[0,1].plot(n1Poblacion,n2Poblacion)
    axs2[0,1].scatter(20,20,label="Condiciones iniciales", color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 60, 80, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs2[0,1].plot(n1Poblacion,n2Poblacion)
    axs2[0,1].scatter(60,80,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 45, 45, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs2[0,1].plot(n1Poblacion,n2Poblacion)
    axs2[0,1].scatter(45,45,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 30, 40, r1, r2, 100, 50, 1, 1, h, num_steps)
    axs2[0,1].plot(n1Poblacion,n2Poblacion)
    axs2[0,1].scatter(30,40,color="black")
    axs2[0,1].plot(0, 0, 'ko', markersize=0.0001, label='K1=100, K2=50, α12=1, α21=1')

    #Puntos
    estabilidadPuntos(K1=100, K2=50, alpha12=1, alpha21=1, N1=0, N2=50, caso = "Caso 2 - Extinción especie 1")
    axs2[0,1].plot(100, 0, 'ko', markersize=7, label='Puntos de equilibrio inestable', color="#FFD700")
    estabilidadPuntos(K1=100, K2=50, alpha12=1, alpha21=1, N1=100, N2=0, caso = "Caso 2 - Extinción especie 2")
    axs2[0,1].plot(0, 50, 'ko', markersize=7, label='Puntos de equilibrio estable', color="grey")

    plot_isoclinas(axs2[1,0],70, 100, 1, 2,"Caso3 - Equilibrio inestable")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 20, 20, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs2[1,0].plot(n1Poblacion,n2Poblacion)
    axs2[1,0].scatter(20,20,label="Condiciones iniciales",color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 10, 80, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs2[1,0].plot(n1Poblacion,n2Poblacion,)
    axs2[1,0].scatter(10,80,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 60, 60, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs2[1,0].plot(n1Poblacion,n2Poblacion)
    axs2[1,0].scatter(60,60,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 50, 10, r1, r2, 70, 100, 1, 2, h, num_steps)
    axs2[1,0].plot(n1Poblacion,n2Poblacion)
    axs2[1,0].scatter(50,10,color="black")
    axs2[1,0].plot(0, 0, 'ko', markersize=0.0001, label='K1=70, K2=100, α12=1, α21=2')

    #Puntos
    estabilidadPuntos(K1=70, K2=100, alpha12=1, alpha21=2, N1=0, N2=100, caso = "Caso 3 - Extincion 1")
    axs2[1,0].plot(0, 100, 'ko', markersize=7, label='Puntos de equilibrio estable', color="#FFD700")
    estabilidadPuntos(K1=70, K2=100, alpha12=1, alpha21=2, N1=70, N2=0, caso = "Caso 3 - Extincion 2")
    axs2[1,0].plot(70, 0, 'ko', markersize=7, color="#FFD700")
    estabilidadPuntos(K1=70, K2=100, alpha12=1, alpha21=2, N1=30, N2=40, caso = "Caso 3 - Equilibrio")
    axs2[1,0].plot(30, 40, 'ko', markersize=7, label='Puntos de equilibrio inestable', color="grey")


    plot_isoclinas(axs2[1,1],75, 50, 1, .5,"Caso4 - Coexistencia estable")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 20, 20, r1, r2, 75, 50, 1, .5, h, num_steps)
    axs2[1,1].plot(n1Poblacion,n2Poblacion)
    axs2[1,1].scatter(20,20,label="Condiciones iniciales", color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 10, 60, r1, r2, 75, 50, 1, .5, h, num_steps)
    axs2[1,1].plot(n1Poblacion,n2Poblacion)
    axs2[1,1].scatter(10,60,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 80, 10, r1, r2, 75, 50, 1, .5, h, num_steps)
    axs2[1,1].plot(n1Poblacion,n2Poblacion)
    axs2[1,1].scatter(80,10,color="black")
    n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, 70, 70, r1, r2, 75, 50, 1, .5, h, num_steps)
    axs2[1,1].plot(n1Poblacion,n2Poblacion)
    axs2[1,1].scatter(70,70,color="black")
    axs2[1,1].plot(0, 0, 'ko', markersize=0.0001, label='K1=75, K2=50, α12=1, α21=0.5')
    
    
    #Puntos
    estabilidadPuntos(K1=75, K2=50, alpha12=1, alpha21=.5, N1=50, N2=50, caso = "Caso 4 - Coexistencia")
    axs2[1,1].plot(50, 25, 'ko', markersize=7, label='Puntos de equilibrio estable', color='#FFD700')
    estabilidadPuntos(K1=75, K2=50, alpha12=1, alpha21=.5, N1=75, N2=0, caso = "Caso 4 - Extincion 1")
    axs2[1,1].plot(75, 0, 'ko', markersize=7, label='Puntos de equilibrio inestable', color='grey')
    estabilidadPuntos(K1=75, K2=50, alpha12=1, alpha21=.5, N1=0, N2=50, caso = "Caso 4 - Extincion 2")
    axs2[1,1].plot(0, 50, 'ko', markersize=7, color='grey')

    axs2[0,0].legend(fontsize='small')
    axs2[0,1].legend(fontsize='small')
    axs2[1,1].legend(fontsize='small')
    axs2[1,0].legend(fontsize='small')

    plt.tight_layout()
    plt.show()


def estabilidadPuntos(K1, K2, alpha12, alpha21, N1, N2, caso):
    r1 = 0.3
    r2 = 0.5
    J_evaluada = np.array([
        [r1 * (1 - 2 * N1 / K1 - alpha12 * N2 / K1), -r1 * alpha12 * N1 / K1],
        [-r2 * alpha21 * N2 / K2, r2 * (1 - N2 / K2 - alpha21 * N1 / K2)]])
    autovalores, autovectores = np.linalg.eig(J_evaluada)

    print(caso)
    if any(autovalores > 0):
        print("Punto de equilibrio inestable")
    elif any(autovalores == 0):
        print("Punto de equilibrio neutral")
    else:
        print("Punto de equilibrio estable")
    print(f"autovalores: {autovalores}\n")


def main():
    casos()
    isoclinas_puntos()

if __name__ == "__main__":
    main()