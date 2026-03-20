import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def similaridad(xi, xj, sigma):
    norm_sq = np.linalg.norm(xi - xj) ** 2
    return np.exp(-norm_sq / (2 * sigma ** 2))

def matriz_similaridad(X, sigma, funcion_similaridad=similaridad):
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = funcion_similaridad(X[i], X[j], sigma)
    return S

def reducir_dimension(X, d):  
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Vt_d = Vt[:d, :]
    X_reduced = X@Vt_d.T
    return X_reduced



def grafico_similaridad(X, sigma, ax, funcion_similaridad=similaridad, title='Matriz de similaridad'):
    M = matriz_similaridad(X, sigma, funcion_similaridad=funcion_similaridad)
    im = ax.imshow(M, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

def grafico_similaridad_reducida(X, d, sigma, ax):
    X_reduced = reducir_dimension(X, d)
    grafico_similaridad(X_reduced, sigma, ax, title=f'Similaridad de {d} dimensiones')



def graficar_valores_singulares(s):
    plt.figure(figsize=(10, 6))
    plt.plot(s, 'o-', markersize=8, color='b', label='Valores Singulares')
    plt.title('Magnitudes de los Valores Singulares')
    plt.xlabel('Índice')
    plt.ylabel('Valor Singular')
    plt.grid(True)
    plt.legend()
    plt.show()



def varianza_explicada_por_lista_componentes(s, lista_componentes):
    varianza_explicada = (s ** 2) / np.sum(s ** 2)
    varianza_explicada_acumulada = np.cumsum(varianza_explicada)
    
    colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(s) + 1), varianza_explicada_acumulada, 's-', color='gray', label='Varianza Explicada Acumulada')
    
    for i, num_componentes in enumerate(lista_componentes):
        # Varianza explicada acumulada por el número de componentes en la lista
        varianza_acumulada_d = varianza_explicada_acumulada[:num_componentes]
        # Línea vertical para indicar el número de componentes
        plt.axvline(x=num_componentes, color=colores[i], linestyle='--', label=f'{num_componentes} Componentes', alpha=0.7)
        # Línea horizontal para mostrar la varianza explicada por esos componentes
        plt.axhline(y=varianza_acumulada_d[-1], color=colores[i], linestyle='--', label=f'Varianza Explicada: {varianza_acumulada_d[-1]*100:.2f}%', alpha=0.7)

    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada Acumulada por Número de Componentes')
    plt.legend()
    plt.grid(True)
    plt.show()
def cuadrados_minimos(X, y, threshold=1e-10):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    
    # Reemplazar valores pequeños de s por 0
    s[s < threshold] = 0

    S_inv = np.zeros_like(s)
    S_inv[s != 0] = 1.0 / s[s != 0]
    S_inv = np.diag(S_inv)
    
    svd = vt.T @ S_inv @ u.T

    beta_svd = svd @ y 
    y_pred_svd = X @ beta_svd

    error_total_svd = np.mean(np.linalg.norm(y - y_pred_svd))
    print("Error de predicción total (MSE) usando SVD:", error_total_svd)

    errores = []
    dimensiones = []

    for d in range(1, 105):
        X_reduced = reducir_dimension(X, d)
        
        u_reduced, s_reduced, vt_reduced = np.linalg.svd(X_reduced, full_matrices=False)
        
        # Reemplazar valores pequeños de s por 0
        s_reduced[s_reduced < threshold] = 0

        S_inv_reduced = np.zeros_like(s_reduced)
        S_inv_reduced[s_reduced != 0] = 1.0 / s_reduced[s_reduced != 0]
        S_inv_reduced = np.diag(S_inv_reduced)
        
        SVD = vt_reduced.T @ S_inv_reduced @ u_reduced.T 

        beta_reduced_svd = SVD @ y    
        y_pred_reduced_svd = X_reduced @ beta_reduced_svd

        error_svd = np.mean(np.linalg.norm(y - y_pred_reduced_svd))
        errores.append(error_svd)
        dimensiones.append(d)

    return errores, dimensiones,beta_svd

X = pd.read_csv('dataset.csv')
y = pd.read_csv('y.txt', header=None, delimiter=' ')

X = X.iloc[:, 1:].values  

X_centrado = X - np.mean(X, axis=0)

y = y.values.flatten()
y = y - np.mean(y)

muestra= X_centrado
plt.imshow(muestra, cmap='viridis', interpolation='none')

plt.tight_layout()
plt.colorbar()
plt.title('Matriz de ejemplo')
plt.show()

X_reduced_3d = reducir_dimension(X_centrado, 3)
X_reduced_2d = reducir_dimension(X_centrado, 2)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c='b', marker='o')
ax.set_title('Reducción a 3 dimensiones')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')

ax2 = fig.add_subplot(122)
ax2.scatter(X_reduced_2d[:, 1], X_reduced_2d[:, 0], c='r', marker='o')
ax2.set_title('Reducción a 2 dimensiones')
ax2.set_xlabel('Componente 1')
ax2.set_ylabel('Componente 2')
plt.tight_layout()
plt.show()

sigma = 1

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

grafico_similaridad_reducida(X, 2, sigma, axs[0])
grafico_similaridad_reducida(X, 6, sigma, axs[1])
grafico_similaridad_reducida(X, 10, sigma, axs[2])


plt.show()




#1.2)


U, s, Vt = np.linalg.svd(X_centrado, full_matrices=False)


graficar_valores_singulares(s)


primer_vector_Vt = Vt[0]

# Graficar la importancia de cada dimensión original
plt.bar(range(len(primer_vector_Vt)), np.abs(primer_vector_Vt))
plt.xlabel('Dimensión original')
plt.ylabel('Valor absoluto de la componente')
plt.title('Importancia de cada dimensión según el primer vector de V^T')
plt.show()


varianza_explicada_por_lista_componentes(s, [2, 6, 10,102])


errores, dimensiones,beta_total= cuadrados_minimos(X_centrado, y)



plt.figure(figsize=(10, 6))
plt.plot(dimensiones, errores, marker='o', label='SVD')
plt.xlabel('Dimensión')
plt.ylabel('Error de predicción (MSE)')
plt.title('Error de predicción por dimensión (PCA)')
plt.legend()
plt.grid(True)
plt.show()

X_2reduced = reducir_dimension(X_centrado, 2)

u_2reduced, s_2reduced, vt_2reduced = np.linalg.svd(X_2reduced, full_matrices=False)
s_2reduced[s_2reduced < 1e-10] = 0

S_2inv = np.zeros_like(s_2reduced)
S_2inv[s_2reduced != 0] = 1.0 / s_2reduced[s_2reduced != 0]
S_2inv = np.diag(S_2inv)
beta_2d_svd = vt_2reduced.T @ S_2inv @u_2reduced.T @ y

y_pred_2d_svd = X_2reduced@ beta_2d_svd

error_2d_svd = np.mean(np.linalg.norm(y - y_pred_2d_svd))
print("Error de predicción (MSE) para 2 dimensiones usando SVD:", error_2d_svd)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

X_2d = X_2reduced[:, 0]
Y_2d = X_2reduced[:, 1]

Z_pred = y_pred_2d_svd

Z_real = y

scatter_real = ax.scatter(X_2d, Y_2d, Z_real, c='blue', marker='o', label='Datos Reales')

scatter_pred = ax.scatter(X_2d, Y_2d, Z_pred, c='red', marker='^', label='Predicciones')

ax.set_xlabel('Primera componente principal')
ax.set_ylabel('Segunda componente principal')
ax.set_zlabel('Valor')

plt.title('Datos reducidos a 2 dimensiones con predicciones en 3D')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y, Z_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Comparación de predicciones vs. valores reales (mejor modelo PCA)')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(beta_total)),np.absolute(beta_total) , color='b', label='SVD')
plt.xlabel('dimensiones')
plt.ylabel('valor absoluto de beta')
plt.title('importancia de los valores de beta ')
plt.grid(True)
plt.show()


print("Vector beta para la reducción a 2 dimensiones usando SVD:")
print(beta_2d_svd)






