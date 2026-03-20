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


data = pd.read_csv('dataset2.csv', header=0, index_col=0)




# muestra= reducir_dimension(X_centrado, 102)
# plt.imshow(muestra, cmap='viridis', interpolation='none')

# plt.tight_layout()
# plt.colorbar()
# plt.title('Matriz de ejemplo')
# # plt.show()

# X_reduced_3d = reducir_dimension(X_centrado, 3)
# X_reduced_2d = reducir_dimension(X_centrado, 2)

# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c='b', marker='o')
# ax.set_title('Reducción a 3 dimensiones')
# ax.set_xlabel('Componente 1')
# ax.set_ylabel('Componente 2')
# ax.set_zlabel('Componente 3')

# ax2 = fig.add_subplot(122)
# ax2.scatter(X_reduced_2d[:, 1], X_reduced_2d[:, 0], c='r', marker='o')
# ax2.set_title('Reducción a 2 dimensiones')
# ax2.set_xlabel('Componente 1')
# ax2.set_ylabel('Componente 2')
# plt.tight_layout()
# plt.show()

# sigma = 1

# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# grafico_similaridad_reducida(X, 2, sigma, axs[0])
# grafico_similaridad_reducida(X, 6, sigma, axs[1])
# grafico_similaridad_reducida(X, 10, sigma, axs[2])


# plt.show()




# #1.2)


# U, s, Vt = np.linalg.svd(X_centrado, full_matrices=False)


# graficar_valores_singulares(s)


# primer_vector_Vt = Vt[50]

# # Graficar la importancia de cada dimensión original
# plt.bar(range(len(primer_vector_Vt)), np.abs(primer_vector_Vt))
# plt.xlabel('Dimensión original')
# plt.ylabel('Valor absoluto de la componente')
# plt.title('Importancia de cada dimensión según el primer vector de V^T')
# plt.show()


# varianza_explicada_por_lista_componentes(s, [2, 6, 10,102])

# dimensiones=[]
# errores_svd = []

# u, s, vt = np.linalg.svd(X_centrado, full_matrices=False)

# S_inv = np.diag(1.0 / s)
# svd = vt.T@S_inv@u.T

# beta_svd =svd @ y 
# print(beta_svd)
# y_pred_svd = X_centrado@ beta_svd

# error_total_svd = np.mean(np.linalg.norm(y - y_pred_svd))
# print("Error de predicción total (MSE) usando SVD:", error_total_svd)

# for d in range (1, 105):
#     X_reduced = reducir_dimension(X_centrado, d)
    
#     u_reduced, s_reduced, vt_reduced = np.linalg.svd(X_reduced, full_matrices=False)
#     S_inv_reduced = np.diag(1.0 / s_reduced)
#     SVD = vt_reduced.T @ S_inv_reduced @u_reduced.T 

#     beta_reduced_svd = SVD @ y    
#     y_pred_reduced_svd = X_reduced@ beta_reduced_svd
    
#     error_svd = np.mean(np.linalg.norm(y - y_pred_reduced_svd))
#     errores_svd.append(error_svd)
#     dimensiones.append(d)
# dimensiones.append(105)
# errores_svd.append(error_total_svd)

# plt.figure(figsize=(10, 6))
# plt.plot(dimensiones, errores_svd, marker='o', label='SVD')
# plt.xlabel('Dimensión')
# plt.ylabel('Error de predicción (MSE)')
# plt.title('Error de predicción por dimensión (PCA)')
# plt.legend()
# plt.grid(True)
# plt.show()

# X_2reduced = reducir_dimension(X_centrado, 2)

# u_2reduced, s_2reduced, vt_2reduced = np.linalg.svd(X_2reduced, full_matrices=False)
# S_inv_2reduced = np.diag(1.0 / s_2reduced)
# beta_2d_svd = vt_2reduced.T @ S_inv_2reduced @u_2reduced.T @ y

# y_pred_2d_svd = X_2reduced@ beta_2d_svd

# error_2d_svd = np.mean(np.linalg.norm(y - y_pred_2d_svd))
# print("Error de predicción (MSE) para 2 dimensiones usando SVD:", error_2d_svd)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# X_2d = X_2reduced[:, 0]
# Y_2d = X_2reduced[:, 1]

# Z_pred = y_pred_2d_svd

# Z_real = y

# scatter_real = ax.scatter(X_2d, Y_2d, Z_real, c='blue', marker='o', label='Datos Reales')

# scatter_pred = ax.scatter(X_2d, Y_2d, Z_pred, c='red', marker='^', label='Predicciones')

# ax.set_xlabel('Primera componente principal')
# ax.set_ylabel('Segunda componente principal')
# ax.set_zlabel('Valor')

# plt.title('Datos reducidos a 2 dimensiones con predicciones en 3D')
# plt.legend()
# # plt.show()



# plt.figure(figsize=(10, 6))
# plt.bar(np.arange(len(beta_svd)),np.absolute(beta_svd) , color='b', label='SVD')
# plt.xlabel('dimensiones')
# plt.ylabel('valor de beta')
# plt.title('importancia de los valores de beta ')
# plt.grid(True)
# plt.show()


# print("Vector beta para la reducción a 2 dimensiones usando SVD:")
# print(beta_2d_svd)



# def graficar_punto1c():
#     # Calcular la pseudo inversa de X
#     X_centered = X - np.mean(X, axis=0)
#     X_pinv = np.linalg.pinv(X_centered)
#     Y_centered = y - np.mean(y)
#     U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

#     # Despejar b
#     b = X_pinv @ Y_centered

#     # Calcular la predicción
#     y_pred = X_centered @ b

#     # Calcular el error de predicción
#     error = np.mean(np.linalg.norm(Y_centered - y_pred))


#     # Graficar las predicciones vs. valores reales
#     plt.figure(figsize=(10, 7))
#     plt.scatter(Y_centered, y_pred, alpha=0.5)
#     plt.plot([min(Y_centered), max(Y_centered)], [min(Y_centered), max(Y_centered)], color='red')
#     plt.xlabel('Valores Reales')
#     plt.ylabel('Predicciones')
#     plt.title('Predicciones vs. Valores Reales')
#     plt.grid(True)
#     plt.show()

#     # Graficar los pesos de las dimensiones originales
#     plt.figure(figsize=(10, 7))
#     plt.bar(range(len(b)), abs(b))
#     plt.xlabel('Dimensión Original')
#     plt.ylabel('Peso (β)')
#     plt.title('Pesos de las Dimensiones Originales (valor absoluto)')
#     plt.grid(True)
#     plt.show()

#     # Calcular los errores de predicción
#     errors = np.abs(y - y_pred)

#     # Identificar las muestras con menor error de predicción
#     sorted_indices = np.argsort(errors)
#     best_samples_indices = sorted_indices[:10]  # Selecciona las 10 muestras con menor error, por ejemplo

#     # Graficar los errores de predicción
#     plt.figure(figsize=(10, 6))
#     plt.plot(errors, 'bo', markersize=5, label='Errores de predicción')

#     # Resaltar las mejores muestras
#     plt.plot(best_samples_indices, errors[best_samples_indices], 'ro', markersize=7, label='Mejores muestras')

#     # Añadir etiquetas y título
#     plt.xlabel('Índice de la muestra')
#     plt.ylabel('Error de predicción')
#     plt.title('Errores de Predicción y Mejores Muestras')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Calcular PCA para obtener las tres primeras componentes principales
#     X_pca= reducir_dimension(X_centered, 3)

#     # Crear gráfico 3D
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Graficar las predicciones vs. valores reales usando las componentes principales
#     ax.scatter(X_pca[:, 0], X_pca[:, 1], Y_centered, c='b', marker='o', alpha=0.5, label='Valores Reales')
#     ax.scatter(X_pca[:, 0], X_pca[:, 1], y_pred, c='r', marker='x', alpha=0.5, label='Predicciones')

#     ax.set_xlabel('Componente Principal 1')
#     ax.set_ylabel('Componente Principal 2')
#     ax.set_zlabel('Valores Reales / Predicciones')
#     ax.set_title('Predicciones vs. Valores Reales en 3D')
#     ax.legend()
#     plt.show()

#     # Graficar el error de predicción vs la cantidad de dimensiones
#     errores = []
#     dimensiones = list(range(1, X_centered.shape[1] + 1))
#     for d in dimensiones:
#         X_reducido = reducir_dimension(X_centered, d)
#         X_reducido_pinv = np.linalg.pinv(X_reducido)
#         b_reducido = X_reducido_pinv @ Y_centered
#         y_pred_reducido = X_reducido @ b_reducido
#         error_reducido = np.mean(np.linalg.norm(Y_centered - y_pred_reducido))

#         errores.append(error_reducido)

#     plt.figure(figsize=(10, 7))
#     plt.plot(dimensiones, errores, marker='o')
#     plt.xlabel('Número de Dimensiones')
#     plt.ylabel('Error de Predicción (MSE)')
#     plt.title('Error de Predicción (PCA) vs. Cantidad de Dimensiones')
#     plt.grid(True)
#     plt.show()




def PuntoTres():
    y = pd.read_csv('y.txt', header=None, delimiter=' ').values.flatten()   
    X = data - data.mean(axis=0)
    y = y - y.mean()
    errores = []
    for d in range(1, 207):
        matrix = reducir_dimension(X,d)
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        SInv = np.diag(1 / S)
        Matrix_pseudo = Vt.T @ SInv @ U.T
        solucion = Matrix_pseudo @ y

        error_total = np.linalg.norm((matrix @ solucion) - y)
        errores.append(error_total)

    # Gráfico de errores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 207), errores, color='r')
    plt.title('Error Total de Aproximación en Función de d')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Error Total de Aproximación')
    plt.grid(True)
    plt.show()


    # Gráfica en 2D
    d = 2
    matrix_2d = reducir_dimension(X,d)
    U, S, Vt = np.linalg.svd(matrix_2d, full_matrices=False)
    SInv = np.diag(1 / S)
    Matrix_pseudo = Vt.T @ SInv @ U.T
    solucion_2d = Matrix_pseudo @ y
    aproximacion = matrix_2d @ solucion_2d

    #Original
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    SInv = np.diag(1 / S)
    X_pseudo = Vt.T @ SInv @ U.T
    solucion = X_pseudo @ y

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 207),np.absolute(solucion), color='b')
    plt.title('Peso de dimensiones en la solución original')
    plt.xlabel('Dimensiones')
    plt.ylabel('Peso')
    plt.grid(True)
    plt.show()


PuntoTres()
