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

def plot_population(ax, tiempo, n1Poblacion, n2Poblacion, label):
    ax.plot(tiempo, n1Poblacion, label='Poblacion 1') 
    ax.plot(tiempo, n2Poblacion, label='Poblacion 2') 
    ax.legend()

def plot_isoclinas(ax, K1,K2,alpha12,alpha21):
    N1_values = np.linspace(0, 1.5 * K1, 100)
    N2_values = np.linspace(0, 1.5 * K2, 100)
    N1_grid, N2_grid = np.meshgrid(N1_values, N2_values)
    dN1 = dN1_dt(N1_grid, N2_grid, r1, K1, alpha12)
    dN2 = dN2_dt(N1_grid, N2_grid, r2, K2, alpha21)
    ax.contour(N1_grid, N2_grid, dN1, levels=[0], colors='red')
    ax.contour(N1_grid, N2_grid, dN2, levels=[0], colors='brown')

    # Calculamos las variaciones de la población en cada punto de la cuadrícula
    dN1 = dN1_dt(N1_grid, N2_grid, r1, K1, alpha12)
    dN2 = dN2_dt(N1_grid, N2_grid, r2, K2, alpha21)

    # Trazamos el campo vectorial usando streamplot
    ax.streamplot(N1_grid, N2_grid, dN1, dN2, density=.5, arrowsize=1, color='blue')


tiempo = np.linspace(0, 100, 400)
N1_0 = 1
N2_0 = 1  
r1 = 0.3
r2 = 0.2 
h = 1    
num_steps = 399 


fig, axs = plt.subplots(2,2, figsize=(18, 6))  # 2 fila, 2 columnas


    
n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, N1_0, N2_0, r1, r2, 50, 100, 1, 1, h, num_steps)
plot_population(axs[0,0], tiempo, n1Poblacion, n2Poblacion, f'K1={50}, K2={100}, alpha12={1}, alpha21={1}')

n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, N1_0, N2_0, r1, r2, 100, 50, 1, 1, h, num_steps)
plot_population(axs[0,1], tiempo, n1Poblacion, n2Poblacion, f'K1={100}, K2={50}, alpha12={1}, alpha21={1}')

n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, N1_0, N2_0, r1, r2, 70, 100, 1, 2, h, num_steps)
plot_population(axs[1,0], tiempo, n1Poblacion, n2Poblacion, f'K1={70}, K2={100}, alpha12={1}, alpha21={2}')

n1Poblacion, n2Poblacion = runge_kutta_4(dN1_dt, dN2_dt, N1_0, N2_0, r1, r2, 75, 50, 1, .5, h, num_steps)
plot_population(axs[1,1], tiempo, n1Poblacion, n2Poblacion, f'K1={75}, K2={50}, alpha12={1}, alpha21={.5}')


fig.suptitle('Tiempo VS Poblacion', fontsize=16)


plt.tight_layout()
plt.show()
fig2, axs2 = plt.subplots(2,2, figsize=(18, 6))  # 2 fila, 2 columnas
fig2.suptitle('Curvas Isoclinas y Campo Vectorial', fontsize=16)

plot_isoclinas(axs2[0,0], 50, 100, 1, 1)
plot_isoclinas(axs2[0,1],100, 50, 1, 1)
plot_isoclinas(axs2[1,0],70, 100, 1, 2)
plot_isoclinas(axs2[1,1],75, 50, 1, .5)
plt.tight_layout()
plt.show()


