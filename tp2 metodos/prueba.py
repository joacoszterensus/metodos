import matplotlib.pyplot as plt

# Datos para los gráficos
x = [0, 10]
y_vertical = [0, 10]
y_horizontal = [5, 5]

# Crear la figura y los ejes para los gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico con la línea vertical
ax1.plot([5, 5], y_vertical, 'b-')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_xlabel('Densidad de Presas')
ax1.set_ylabel('Densidad de Depredadores')
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)  # Quitar las escalas
ax1.grid(False)  # Desactivar la cuadrícula
ax1.spines['left'].set_position('zero')
ax1.spines['bottom'].set_position('zero')
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.spines['left'].set_linewidth(2)  # Hacer la espina izquierda más gruesa
ax1.spines['bottom'].set_linewidth(2)  # Hacer la espina inferior más gruesa

# Gráfico con la línea horizontal
ax2.plot(x, y_horizontal, 'r-')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_xlabel('Densidad de Presas')
ax2.set_ylabel('Densidad de Depredadores')
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)  # Quitar las escalas
ax2.grid(False)  # Desactivar la cuadrícula
ax2.spines['left'].set_position('zero')
ax2.spines['bottom'].set_position('zero')
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.spines['left'].set_linewidth(2)  # Hacer la espina izquierda más gruesa
ax2.spines['bottom'].set_linewidth(2)  # Hacer la espina inferior más gruesa

# Mostrar los gráficos
plt.tight_layout()
plt.show()
