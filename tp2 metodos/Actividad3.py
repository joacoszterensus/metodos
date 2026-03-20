import numpy as np
import matplotlib.pyplot as plt

def dN_dt(n,p,r,alpha):
    return r*n-alpha*n*p

def dP_dt(n,p,beta,q):
    return beta*n*p-q*p

def dN_dt_LVE(n,k,p,r,alpha):
    return r*n*(1-n/k)-alpha*n*p

def runge_kutta_4(dN_dt, dP_dt, n0, p0, r,q, alpha, beta, dt, steps):

    n_values = [n0]
    p_values = [p0]
    
    for i in range(1, steps+1):
        k1_n = dN_dt(n_values[-1], p_values[-1], r, alpha)
        k1_p = dP_dt(n_values[-1], p_values[-1], beta, q)
        
        k2_n = dN_dt(n_values[-1] + 0.5*dt*k1_n, p_values[-1] + 0.5*dt*k1_p, r, alpha)
        k2_p = dP_dt(n_values[-1] + 0.5*dt*k1_n, p_values[-1] + 0.5*dt*k1_p, beta, q)
        
        k3_n = dN_dt(n_values[-1] + 0.5*dt*k2_n, p_values[-1] + 0.5*dt*k2_p, r, alpha)
        k3_p = dP_dt(n_values[-1] + 0.5*dt*k2_n, p_values[-1] + 0.5*dt*k2_p, beta, q)
        
        k4_n = dN_dt(n_values[-1] + dt*k3_n, p_values[-1] + dt*k3_p, r, alpha)
        k4_p = dP_dt(n_values[-1] + dt*k3_n, p_values[-1] + dt*k3_p, beta, q)
        
        n_new = n_values[-1] + (dt/6)*(k1_n + 2*k2_n + 2*k3_n + k4_n)
        p_new = p_values[-1] + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        n_values.append(n_new)
        p_values.append(p_new)
    
    return n_values, p_values

def runge_kutta_4_2(dN_dt, dP_dt, n0, p0, r,q, alpha, beta,k, dt, steps):
    n_values = [n0]
    p_values = [p0]
    
    for i in range(1, steps+1):
        k1_n = dN_dt(n_values[-1],k, p_values[-1], r, alpha)
        k1_p = dP_dt(n_values[-1], p_values[-1], beta, q)
        
        k2_n = dN_dt(n_values[-1] + 0.5*dt*k1_n,k, p_values[-1] + 0.5*dt*k1_p, r, alpha)
        k2_p = dP_dt(n_values[-1] + 0.5*dt*k1_n, p_values[-1] + 0.5*dt*k1_p, beta, q)
        
        k3_n = dN_dt(n_values[-1] + 0.5*dt*k2_n,k, p_values[-1] + 0.5*dt*k2_p, r, alpha)
        k3_p = dP_dt(n_values[-1] + 0.5*dt*k2_n, p_values[-1] + 0.5*dt*k2_p, beta, q)
        
        k4_n = dN_dt(n_values[-1] + dt*k3_n,k, p_values[-1] + dt*k3_p, r, alpha)
        k4_p = dP_dt(n_values[-1] + dt*k3_n, p_values[-1] + dt*k3_p, beta, q)
        
        n_new = n_values[-1] + (dt/6)*(k1_n + 2*k2_n + 2*k3_n + k4_n)
        p_new = p_values[-1] + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        n_values.append(n_new)
        p_values.append(p_new)
    
    return n_values, p_values

def plot_population(ax, tiempo, n1Poblacion, n2Poblacion, label):
    ax.plot(tiempo, n1Poblacion, label=label+" Presa") 
    ax.plot(tiempo, n2Poblacion, label=label+" depredador") 
    ax.legend()

def isoclinas_DN_DT(r,alpha):
    return r/alpha
def isoclinas_DP_DT(q,beta):
    return q/beta
def isoclinas_DN_DT_LVE(k,r,alpha,n):
    return r*(1-n/k)/alpha

def plot_isoclinas(ax, r, alpha, q, beta, label):
    values = np.linspace(0, 100, 100)
    
    ax.hlines(y=isoclinas_DN_DT(r, alpha), xmin=min(values), xmax=max(values), label=label)
    ax.vlines(x=isoclinas_DP_DT(q, beta), ymin=min(values), ymax=max(values))
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()

def plot_isoclinas2(ax, k, r, alpha, q, beta, label):
    values = np.linspace(0, 100, 400)
    
    # Calcula las isoclinas para dN/dt
    isoclinas_N = isoclinas_DN_DT_LVE(k,r,alpha, values)
    

    # Grafica las isoclinas para dN/dt
    ax.plot(values, isoclinas_N, label=label)
    
    ax.vlines(x=isoclinas_DP_DT(q, beta), ymin=min(values), ymax=max(values))

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()



tiempo = np.linspace(0, 400, 400)
n=50
p=20
r=0.1
q=0.1
alpha=0.005
beta=0.004
k=50
h=1
steps=399




fig, axs = plt.subplots(2,2, figsize=(18, 6))  
    
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,n,p,r,q,alpha,beta,h,steps)
plot_population(axs[0,0], tiempo, n1Poblacion, n2Poblacion, f'Ecuaciones normales')

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,n,p,r,q,alpha,beta,k,h,steps)
plot_population(axs[0,1], tiempo, n1Poblacion, n2Poblacion, f'Ecuaciones Extendidas')


n1=200
p1=10
r1=0.05
q1=0.15
alpha1=0.002
beta1=0.002
k1=300

n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,n1,p1,r1,q1,alpha1,beta1,h,steps)
plot_population(axs[1,0], tiempo, n1Poblacion, n2Poblacion, f'Ecuaciones normales')

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,n1,p1,r1,q1,alpha1,beta1,k1,h,steps)
plot_population(axs[1,1], tiempo, n1Poblacion, n2Poblacion, f'Ecuaciones Extendidas')



fig.suptitle('Tiempo VS Poblacion', fontsize=16)

plt.tight_layout()
plt.show()


#isoclinas


fig2, axs2 = plt.subplots(2,2, figsize=(18, 6))  # 2 fila, 2 columnas
fig2.suptitle('Curvas Isoclinas y Campo Vectorial', fontsize=16)

plot_isoclinas(axs2[0,0],r,alpha,q,beta,"isoclinicas no extendidas")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,100,10,r,q,alpha,beta,h,steps)
axs2[0,0].plot(n1Poblacion,n2Poblacion,label="dep=10,pre=100",color="red")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,80,5,r,q,alpha,beta,h,steps)
axs2[0,0].plot(n1Poblacion,n2Poblacion,label="dep=5,pre=80",color="blue")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,60,3,r,q,alpha,beta,h,steps)
axs2[0,0].plot(n1Poblacion,n2Poblacion,label="dep=3,pre=60",color="green")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,20,1,r,q,alpha,beta,h,steps)
axs2[0,0].plot(n1Poblacion,n2Poblacion,label="dep=1,pre=20",color="brown")
axs2[0,0].legend()

plot_isoclinas(axs2[1,0],r1,alpha1,q1,beta1,"isoclinicas no extendidas")

n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,100,10,r1,q1,alpha1,beta1,h,steps)
axs2[1,0].plot(n1Poblacion,n2Poblacion,label="dep=10,pre=100",color="red")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,50,5,r1,q1,alpha1,beta1,h,steps)
axs2[1,0].plot(n1Poblacion,n2Poblacion,label="dep=5,pre=50",color="blue")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,20,2,r1,q1,alpha1,beta1,h,steps)
axs2[1,0].plot(n1Poblacion,n2Poblacion,label="dep=2,pre=20",color="green")
n1Poblacion, n2Poblacion = runge_kutta_4(dN_dt, dP_dt,200,15,r1,q1,alpha1,beta1,h,steps)
axs2[1,0].plot(n1Poblacion,n2Poblacion,label="dep=15,pre=200",color="brown")
axs2[1,0].legend()
axs2[1,0].set_ylim(bottom=0, top=50)





plot_isoclinas2(axs2[0,1],k,r,alpha,q,beta,"isoclinicas extendidas")
n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,n,p,r,q,alpha,beta,k,h,steps)
axs2[0,1].plot(n1Poblacion,n2Poblacion)
axs2[0,1].scatter(n,p,color="black",label="condiciones iniciales")

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,90,50,r,q,alpha,beta,k,h,steps)
axs2[0,1].plot(n1Poblacion,n2Poblacion)
axs2[0,1].scatter(90,50,color="black")

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,10,10,r,q,alpha,beta,k,h,steps)
axs2[0,1].plot(n1Poblacion,n2Poblacion)
axs2[0,1].scatter(10,10,color="black")



plot_isoclinas2(axs2[1,1],k1,r1,alpha1,q1,beta1,"isoclinicas extendidas")
n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,100,10,r1,q1,alpha1,beta1,k1,h,steps)
axs2[1,1].plot(n1Poblacion,n2Poblacion)
axs2[1,1].scatter(100,10,color="black",label="condiciones iniciales")

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,10,50,r1,q1,alpha1,beta1,k1,h,steps)
axs2[1,1].plot(n1Poblacion,n2Poblacion)
axs2[1,1].scatter(10,50,color="black")

n1Poblacion, n2Poblacion = runge_kutta_4_2(dN_dt_LVE, dP_dt,70,40,r1,q1,alpha1,beta1,k1,h,steps)
axs2[1,1].plot(n1Poblacion,n2Poblacion)
axs2[1,1].scatter(70,40,color="black")


plt.tight_layout()
plt.show()

