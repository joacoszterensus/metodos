
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import griddata

def fb(x1, x2):
    return 0.75*np.e**(-((10*x1-2)**2/4)-((9*x2-2)**2/4)) +  0.65*np.e**(-((9*x1+1)**2/9)-((10*x2+1)**2/2)) + 0.55*np.e**(-((9*x1-6)**2/4)-((9*x2-3)**2/4)) - 0.01*np.e**(-((9*x1-7)**2/4)-((9*x2-3)**2/4))

def chebyshev_points(n):
    k = np.arange(1, n + 1)
    x_k = np.cos((2 * k - 1) * np.pi / (2 * n))
    return x_k

def lagrange_interp_3d(x_values, y_values, z_values, x_interp, y_interp):
    # Número de puntos
    n = len(x_values)

    # Inicializa el arreglo para almacenar los valores interpolados
    z_interp = np.zeros_like(x_interp)

    # Itera sobre los puntos de interpolación
    for i in range(len(x_interp)):
        for j in range(len(y_interp)):
            x = x_interp[i, j]
            y = y_interp[i, j]

            # Inicializa el valor interpolado en el punto (x, y)
            z_interp[i, j] = 0

            # Calcula el polinomio de Lagrange para el punto (x, y)
            for k in range(n):
                num = 1
                den = 1
                for l in range(n):
                    if l != k:
                        num *= (x - x_values[l]) * (y - y_values[l])
                        den *= (x_values[k] - x_values[l]) * (y_values[k] - y_values[l])
                z_interp[i, j] += z_values[k, k] * num / den

    return z_interp



# Genera puntos equiespaciados en el rango de -1 a 1 en ambas variables para la función original
num_points_original = 100
x1_original = np.linspace(-1, 1, num_points_original)
x2_original = np.linspace(-1, 1, num_points_original)
x1_mesh_original, x2_mesh_original = np.meshgrid(x1_original, x2_original)

# Puntos donde deseas interpolar con un espaciado diferente
num_points_interpolated = 9  # 9 puntos equiespaciados cada 0.25 en el rango de -1 a 1
x1_values_interpolated = np.linspace(-1, 1, num_points_interpolated)
x2_values_interpolated = np.linspace(-1, 1, num_points_interpolated)
x1_mesh_interpolated, x2_mesh_interpolated = np.meshgrid(x1_values_interpolated, x2_values_interpolated)

# Calcula los valores de la función en los puntos interpolados
z_values_interpolated = fb(x1_mesh_interpolated, x2_mesh_interpolated)

# Realiza la interpolación en los puntos de interés
interp_values_linear = griddata((x1_mesh_interpolated.ravel(), x2_mesh_interpolated.ravel()), z_values_interpolated.ravel(), (x1_mesh_original, x2_mesh_original), method='linear')
interp_values_splines= griddata((x1_mesh_interpolated.ravel(), x2_mesh_interpolated.ravel()), z_values_interpolated.ravel(), (x1_mesh_original, x2_mesh_original), method='cubic')
interp_values_lagrange = lagrange_interp_3d(x1_values_interpolated, x2_values_interpolated, z_values_interpolated, x1_mesh_original, x2_mesh_original)







##no equiespaciados
# Puntos donde deseas interpolar con un espaciado diferente
num_points_interpolated = 9  # 9 puntos equiespaciados cada 0.25 en el rango de -1 a 1
x1_interpolated_chevy = chebyshev_points(num_points_interpolated)
x2_Interpolated_chevy = chebyshev_points(num_points_interpolated)
x1_mesh_inter_chevy, x2_mesh_inter_chevy = np.meshgrid(x1_interpolated_chevy,x2_Interpolated_chevy)

# Calcula los valores de la función en los puntos interpolados
zValuesInter_chevy = fb(x1_mesh_inter_chevy, x2_mesh_inter_chevy)

# Realiza la interpolación en los puntos de interés
interp_values_linear_chevy = griddata((x1_mesh_inter_chevy.ravel(), x2_mesh_inter_chevy.ravel()), zValuesInter_chevy.ravel(), (x1_mesh_original, x2_mesh_original), method='linear')
interp_values_splines_chevy = griddata((x1_mesh_inter_chevy.ravel(), x2_mesh_inter_chevy.ravel()), zValuesInter_chevy.ravel(), (x1_mesh_original, x2_mesh_original), method='cubic')
inertp_values_lagrange_chevy = lagrange_interp_3d(x1_interpolated_chevy,x2_Interpolated_chevy, zValuesInter_chevy, x1_mesh_original, x2_mesh_original)


real_values = fb(x1_mesh_original, x2_mesh_original)

absolute_error_linear_chevy = np.abs(interp_values_linear_chevy - real_values)


absolute_error_cubic_chevy = np.abs(interp_values_splines_chevy - real_values)


absolute_error_Lagrange_chevy = np.abs(inertp_values_lagrange_chevy - real_values)


absolute_error_linear = np.abs(interp_values_linear - real_values)


absolute_error_cubic = np.abs(interp_values_splines - real_values)


absolute_error_Lagrange = np.abs(interp_values_lagrange - real_values)



fig = plt.figure(figsize=(15, 8))

ax1 = fig.add_subplot(241, projection='3d')
surf1 = ax1.plot_surface(x1_mesh_original, x2_mesh_original, fb(x1_mesh_original, x2_mesh_original), cmap='viridis')
ax1.set_title('Función Original')

ax2 = fig.add_subplot(242, projection='3d')
surf2 = ax2.plot_surface(x1_mesh_original, x2_mesh_original, interp_values_linear, cmap='plasma')
ax2.set_title('Interpolación Lineal')

ax3 = fig.add_subplot(243, projection='3d')
surf3 = ax3.plot_surface(x1_mesh_original, x2_mesh_original, interp_values_linear_chevy, cmap='coolwarm')
ax3.set_title('Interpolación lineal chevysheb')

ax4 = fig.add_subplot(245, projection='3d')
surf4 = ax4.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_linear, cmap='twilight')
ax4.set_title('error absoluto')
ax4.set_zlim(0, 0.6)


ax5 = fig.add_subplot(246, projection='3d')
surf5 = ax5.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_linear_chevy, cmap='twilight')
ax5.set_title('error absoluto chevysheb')
ax5.set_zlim(0, 0.6)





plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(15, 8))

ax1 = fig.add_subplot(241, projection='3d')
surf1 = ax1.plot_surface(x1_mesh_original, x2_mesh_original, fb(x1_mesh_original, x2_mesh_original), cmap='viridis')
ax1.set_title('Función Original')

ax2 = fig.add_subplot(242, projection='3d')
surf2 = ax2.plot_surface(x1_mesh_original, x2_mesh_original, interp_values_lagrange, cmap='plasma')
ax2.set_title('Interpolación lagrange')
ax2.set_zlim(0, 1)

ax3 = fig.add_subplot(243, projection='3d')
surf3 = ax3.plot_surface(x1_mesh_original, x2_mesh_original, inertp_values_lagrange_chevy, cmap='coolwarm')
ax3.set_title('Interpolación  lagrange chevysheb')

ax4 = fig.add_subplot(245, projection='3d')
surf4 = ax4.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_Lagrange, cmap='twilight')
ax4.set_title('error absoluto')
ax4.set_zlim(0, 9)


ax5 = fig.add_subplot(246, projection='3d')
surf5 = ax5.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_Lagrange_chevy, cmap='twilight')
ax5.set_title('error absoluto chevysheb')
ax5.set_zlim(0, 9)




plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(15, 8))

ax1 = fig.add_subplot(241, projection='3d')
surf1 = ax1.plot_surface(x1_mesh_original, x2_mesh_original, fb(x1_mesh_original, x2_mesh_original), cmap='viridis')
ax1.set_title('Función Original')

ax2 = fig.add_subplot(242, projection='3d')
surf2 = ax2.plot_surface(x1_mesh_original, x2_mesh_original, interp_values_splines, cmap='plasma')
ax2.set_title('Interpolación splines')
ax2.set_zlim(0, 1)

ax3 = fig.add_subplot(243, projection='3d')
surf3 = ax3.plot_surface(x1_mesh_original, x2_mesh_original, interp_values_splines_chevy, cmap='coolwarm')
ax3.set_title('Interpolación  splines chevysheb')

ax4 = fig.add_subplot(245, projection='3d')
surf4 = ax4.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_cubic, cmap='twilight')
ax4.set_title('error absoluto')
ax4.set_zlim(0, 0.6)


ax5 = fig.add_subplot(246, projection='3d')
surf5 = ax5.plot_surface(x1_mesh_original, x2_mesh_original, absolute_error_cubic_chevy, cmap='twilight')
ax5.set_title('error absoluto chevysheb')
ax5.set_zlim(0, 0.6)



plt.tight_layout()
plt.show()