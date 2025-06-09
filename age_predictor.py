import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Validar argumentos
if len(sys.argv) < 2:
    print("Uso: python3 age_predictor.py <ruta_a_folder_con_imagenes>")
    sys.exit(1)

# Argumento recibido
FOLDER_PATH = sys.argv[1]  # Ruta desde la línea de comandos

# Constantes
IMG_SIZE = 64
NUM_IMAGES = 6
CATEGORIAS = ['0-9', '10-19', '20-29', '30-39', '40-49', '50+']
MODEL_PATH = "age_model.h5"

# Verificar modelo
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo no encontrado: {MODEL_PATH}")
    sys.exit(1)

# Verificar folder
if not os.path.exists(FOLDER_PATH):
    print(f"❌ Carpeta no encontrada: {FOLDER_PATH}")
    sys.exit(1)

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Obtener lista de imágenes
imagenes = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.jpg')]
if len(imagenes) < NUM_IMAGES:
    print("❌ No hay suficientes imágenes para mostrar.")
    sys.exit(1)

# Elegir imágenes aleatorias
seleccionadas = random.sample(imagenes, NUM_IMAGES)

# Crear figura 2x3
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Predicciones del modelo (categorías de edad)", fontsize=16)

for i, filename in enumerate(seleccionadas):
    path = os.path.join(FOLDER_PATH, filename)
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    input_img = np.expand_dims(img_resized, axis=0)

    # Predicción
    pred = model.predict(input_img, verbose=0)
    categoria = np.argmax(pred)

    # Mostrar imagen y predicción
    ax = axs[i // 3][i % 3]
    ax.imshow(img_rgb)
    ax.set_title(f"Pred: {CATEGORIAS[categoria]}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
