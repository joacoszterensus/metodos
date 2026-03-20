import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def error_porcentaje_norma_frobenius(A, A_reconstruida):
    """Calcula la norma de Frobenius de una matriz A."""
    return np.linalg.norm(A - A_reconstruida, 'fro') / np.linalg.norm(A, 'fro')

def cargar_imagenes(imagen_dir):
    """Carga las imágenes de un directorio y devuelve una matriz de datos aplanada y la lista de nombres de archivos."""
    if not os.path.exists(imagen_dir):
        raise FileNotFoundError(f"La ruta especificada {imagen_dir} no existe.")

    imagenes_archivos = [f for f in os.listdir(imagen_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    imagenes_archivos.sort()
    n = len(imagenes_archivos)
    
    if n == 0:
        raise ValueError(f"No se encontraron imágenes en el directorio {imagen_dir}.")

    # Leer la primera imagen para obtener las dimensiones
    imagen_prueba = Image.open(os.path.join(imagen_dir, imagenes_archivos[0]))
    p = imagen_prueba.size[0]  # Asumiendo que las imágenes son cuadradas, p = ancho = alto

    # Inicializar la matriz de datos
    datos_matriz = np.zeros((n, p * p))

    # Procesar cada imagen
    for i, archivo in enumerate(imagenes_archivos):
        try:
            # Cargar la imagen
            imagen = Image.open(os.path.join(imagen_dir, archivo)).convert('L')  # Convertir a escala de grises
            # Aplanar la imagen y convertirla a un vector
            vector_imagen = np.array(imagen).flatten()
            # Insertar el vector en la matriz de datos
            datos_matriz[i, :] = vector_imagen
        except Exception as e:
            print(f"Error al procesar la imagen {archivo}: {e}")

    return datos_matriz, p, n, imagenes_archivos

def svd_d(matriz, d):
    U, S, Vt = np.linalg.svd(matriz, full_matrices=False)
    U_reducido = U[:, :d]
    S_reducido = np.diag(S[:d])
    Vt_reducido = Vt[:d, :]
    return U_reducido, S_reducido, Vt_reducido

def compresion_svd(datos_matriz, d):
    """Aplica SVD a la matriz de datos y reduce la dimensionalidad a d componentes."""
    U_reducido, S_reducido, Vt_reducido = svd_d(datos_matriz, d)
    datos_reconstruidos = np.dot(U_reducido, np.dot(S_reducido, Vt_reducido))
    return datos_reconstruidos

def reconstruir_matrices(datos_reconstruidos, p, n):
    """Reconstruye las matrices a partir de la matriz de datos reconstruidos."""
    matrices_reconstruidas = []
    for i in range(n):
        try:
            # Convertir el vector reconstruido a una matriz de p x p
            matriz_reconstruida = datos_reconstruidos[i, :].reshape((p, p))
            # Normalizar los valores a rango 0-255
            matriz_reconstruida = np.clip(matriz_reconstruida, 0, 255)
            matrices_reconstruidas.append(matriz_reconstruida)
        except Exception as e:
            print(f"Error al reconstruir la matriz para el índice {i}: {e}")
    return matrices_reconstruidas

def guardar_imagenes(matrices_reconstruidas, reconstruido_dir, imagenes_archivos):
    """Convierte las matrices a imágenes y las guarda en el directorio especificado."""
    os.makedirs(reconstruido_dir, exist_ok=True)

    for i, matriz_reconstruida in enumerate(matrices_reconstruidas):
        try:
            # Convertir la matriz a una imagen PIL
            imagen_reconstruida_pil = Image.fromarray(matriz_reconstruida.astype(np.uint8))
            # Guardar la imagen usando el nombre original
            nombre_original = os.path.basename(imagenes_archivos[i])
            imagen_reconstruida_pil.save(os.path.join(reconstruido_dir, nombre_original))
        except Exception as e:
            print(f"Error al guardar la imagen {nombre_original}: {e}")

    print(f"Las imágenes reconstruidas se han guardado en el directorio '{reconstruido_dir}'")

def reconstruir_y_guardar_imagenes(datos_reconstruidos, p, n, reconstruido_dir, imagenes_archivos):
    """Engloba la reconstrucción de matrices y el guardado de imágenes."""
    matrices_reconstruidas = reconstruir_matrices(datos_reconstruidos, p, n)
    guardar_imagenes(matrices_reconstruidas, reconstruido_dir, imagenes_archivos)

def primer_dataset():
    # Usar una ruta relativa para acceder a la carpeta de imágenes
    imagen_dir = "./datasets_imgs"
    reconstruido_dir = "./imagenes_reconstruidas"
    d = 28 # Número de dimensiones a mantener

    datos_matriz, p, n, imagenes_archivos = cargar_imagenes(imagen_dir)
    U_red, S_red, Vt_red = svd_d(datos_matriz, d)
    i1, i2 = Vt_red[0, :], Vt_red[1, :]
    i1 = np.clip(i1, 0, 255).astype(np.uint8)
    i2 = np.clip(i2, 0, 255).astype(np.uint8)

    # Convertir los vectores a matrices p x p
    I1_matriz = i1.reshape((p, p))
    I2_matriz = i2.reshape((p, p))

    # Mostrar las imágenes usando matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(I1_matriz, cmap='gray')
    plt.title('Imagen 1')

    plt.subplot(1, 2, 2)
    plt.imshow(I2_matriz, cmap='gray')
    plt.title('Imagen 2')

    plt.show()

    datos_reconstruidos = compresion_svd(datos_matriz, d)
    reconstruir_y_guardar_imagenes(datos_reconstruidos, p, n, reconstruido_dir, imagenes_archivos)
    print(p)

def distancia_euclidiana(vec1, vec2, sigma):
    return np.linalg.norm(vec1 - vec2)

def grafico_similaridad(datos_matriz, sigma, show=True):
    n = datos_matriz.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.exp(-np.linalg.norm(datos_matriz[i] - datos_matriz[j]) ** 2 / (2 * sigma ** 2))
    plt.imshow(M, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if show:
        plt.show()

def grafico_similaridad_imagenes(imagen_dir, sigma):
    datos_matriz, p, n, imagenes_archivos = cargar_imagenes(imagen_dir)
    grafico_similaridad(datos_matriz, sigma, show=False)
    plt.xticks(range(n), imagenes_archivos, rotation=90)
    plt.yticks(range(n), imagenes_archivos)
    plt.show()

def grafico_similaridad_reducida_imagenes(imagen_dir, d, sigma):
    datos_matriz, p, n, imagenes_archivos = cargar_imagenes(imagen_dir)
    datos_reducidos = compresion_svd(datos_matriz, d)
    grafico_similaridad(datos_reducidos, sigma, show=False)
    plt.xticks(range(n), imagenes_archivos, rotation=90)
    plt.yticks(range(n), imagenes_archivos)
    plt.show()

def aprender_base_d():
    imagen_dir = "./datasets_imgs_02"
    reconstruido_dir = "./imagenes_reconstruidas_02"
    d = 1 # Número de dimensiones a mantener
    while True:
        datos_matriz, p, n, imagenes_archivos = cargar_imagenes(imagen_dir)
        datos_reconstruidos = compresion_svd(datos_matriz, d)
        error = error_porcentaje_norma_frobenius(datos_matriz, datos_reconstruidos)
        if error <= 0.1:
            break
        d += 1
    print(f"El número mínimo de dimensiones es {d}")
    U_reducido, S_reducido, Vt_reducido = svd_d(datos_matriz, d)
    base_aprendida = np.dot(Vt_reducido.T, Vt_reducido) 
    return base_aprendida

def reconstruir_imagenes_con_base(base_aprendida, imagen_dir, reconstruido_dir):
    datos_matriz, p, n, imagenes_archivos = cargar_imagenes(imagen_dir)
    datos_reconstruidos = np.dot(datos_matriz, base_aprendida)
    reconstruir_y_guardar_imagenes(datos_reconstruidos, p, n, reconstruido_dir, imagenes_archivos)

def graficar_errores_maximos_compresion(errores_maximos):
    plt.bar(range(1, len(errores_maximos)+1), errores_maximos)
    plt.xlabel('Número de dimensiones')
    plt.ylabel('Porcentaje de error')
    plt.title('Error máximo por número de dimensiones')
    plt.show()

if __name__ == "__main__":
    primer_dataset()
