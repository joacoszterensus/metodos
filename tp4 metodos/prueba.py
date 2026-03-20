import numpy as np
import matplotlib.pyplot as plt

def funcionCosto(A, x, b):
    return np.linalg.norm(A @ x - b)**2

def gradienteDescendiente(A, b, x_inicial, step, iter, regularizacion=None, delta=None):
    x = x_inicial
    costo = []
    trayectoria = [x]
    for i in range(iter):
        gradient = 2 * A.T @ (A @ x - b)
        if regularizacion:
            gradient += 2 * delta * x
            x = x - step * gradient
            costo.append(funcionCosto(A, x, b) + delta * np.linalg.norm(x)**2)
        else:
            x = x - step * gradient
            costo.append(funcionCosto(A, x, b))
        trayectoria.append(x)
    return x, costo, trayectoria

def pca_svd(X, n_components=2):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    principal_components = Vt[:n_components]
    X_pca = np.dot(X_centered, principal_components.T)
    return X_pca, principal_components, mean

np.random.seed(0)
n, d = 5, 100
A = np.random.randn(n, d)
b = np.random.randn(n)
x_inicial = np.random.randn(d)

H = 2 * A.T @ A
lambda_max = np.max(np.linalg.eigvals(H).real)
step = 1 / lambda_max

U, S, Vt = np.linalg.svd(A, full_matrices=False)
sigma_max = np.max(S)

x_svd = Vt.T @ np.linalg.pinv(np.diag(S)) @ U.T @ b
costo_svd = funcionCosto(A, x_svd, b)

steps = [0.1, 1, 2]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

trayectoriasF1 = []
final_costs_step = []
xf1 = []
for s in steps:
    actualStep = s / lambda_max
    f1, costs, trayectorias = gradienteDescendiente(A, b, x_inicial, actualStep, 1000)
    xf1.append(f1)
    trayectoriasF1.append(trayectorias)
    final_costs_step.append(costs[-1])
    ax[0].plot(costs, label=f'step={s}/lambda_max')

ax[0].axhline(y=costo_svd, color='black', linestyle='--', label='Solución SVD')
ax[0].set_yscale('log')
ax[0].set_xlabel('Iteración')
ax[0].set_ylabel('Costo (escala logarítmica)')
ax[0].legend()
ax[0].set_title('Costo a lo largo de las iteraciones')
ax[0].grid(True)

ax[1].bar([f'step={s}/lambda_max' for s in steps], final_costs_step)
ax[1].axhline(y=costo_svd, color='black', linestyle='--', label='Solución SVD')
ax[1].set_yscale('log')
ax[1].set_ylabel('Costo final (escala logarítmica)')
ax[1].legend()
ax[1].set_title('Costo final para diferentes tamaños de paso')
ax[1].grid(True)

plt.tight_layout()
plt.show()

deltas = [-10, -2, 1]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
trayectoriasF2 = []
xf2 = []
final_costs_delta = []
for d in deltas:
    actualdelta = (10**d )* (sigma_max)
    f2, costs, trayectorias = gradienteDescendiente(A, b, x_inicial, step, 1000, regularizacion=True, delta=actualdelta)
    xf2.append(f2)
    trayectoriasF2.append(trayectorias)
    final_costs_delta.append(costs[-1])
    ax[0].plot(costs, label=f'delta=10**{d} * sigma_max')
ax[0].set_yscale('log')
ax[0].set_xlabel('Iteración')
ax[0].set_ylabel('Costo (escala logarítmica)')
ax[0].legend()
ax[0].set_title('Costo a lo largo de las iteraciones')
ax[0].grid(True)

ax[1].bar([f'delta=10**{d} * sigma_max' for d in deltas], final_costs_delta)
ax[1].axhline(y=costo_svd, color='black', linestyle='--', label='Solución SVD')
ax[1].set_yscale('log')
ax[1].set_ylabel('Costo final (escala logarítmica)')
ax[1].legend()
ax[1].set_title('Costo final para diferentes parámetros de regularización')
ax[1].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
indices = np.arange(len(x_svd))

plt.plot(indices, x_svd, 'r-', label='Solución SVD')
plt.plot(indices, xf1[1], 'b-', label='Solución F1')
plt.plot(indices, xf2[1], 'g-', label='Solución F2')

plt.xlabel('Índice del vector')
plt.ylabel('Valor del vector')
plt.title('Comparación de vectores finales con condiciones recomendadas')
plt.legend()
plt.grid(True)
plt.show()


A_pca, components, mean = pca_svd(A, n_components=2)
x_inicial_centered = x_inicial - mean
x_inicial_pca = np.dot(x_inicial_centered, components.T)

x1_range = np.linspace(-2, 2, 100)
x2_range = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x_eval = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = funcionCosto(A_pca, x_eval, b)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, tray in enumerate(trayectoriasF1):
    tray_centered = tray - mean
    tray_pca = np.dot(tray_centered, components.T)
    ax = axes[0, i]
    contour = ax.contour(X1, X2, Z, levels=50, cmap='viridis')
    ax.plot(tray_pca[0, 0], tray_pca[0, 1], 'ro', label='Inicio')
    ax.plot(tray_pca[:, 0], tray_pca[:, 1], 'r-', label=f'Trayectoria step={steps[i]}/lambda_max')
    ax.plot(tray_pca[-1, 0], tray_pca[-1, 1], 'bo', label='Fin')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Trayectoria step={steps[i]}/lambda_max')
    ax.legend()
    ax.grid(True)
    fig.colorbar(contour, ax=ax, shrink=0.75)

for i, tray in enumerate(trayectoriasF2):
    tray_centered = tray - mean
    tray_pca = np.dot(tray_centered, components.T)
    ax = axes[1, i]
    contour = ax.contour(X1, X2, Z, levels=50, cmap='viridis')
    ax.plot(tray_pca[0, 0], tray_pca[0, 1], 'ro', label='Inicio')
    ax.plot(tray_pca[:, 0], tray_pca[:, 1], 'r-', label=f'Trayectoria delta=10**{deltas[i]} * sigma_max')
    ax.plot(tray_pca[-1, 0], tray_pca[-1, 1], 'bo', label='Fin')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Trayectoria delta=10**{deltas[i]} * sigma_max')
    ax.legend()
    ax.grid(True)
    fig.colorbar(contour, ax=ax, shrink=0.75)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
