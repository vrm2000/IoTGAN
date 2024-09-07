import time, sys, io
import psutil
from colorama import init, Fore, Style
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from generator import build_generator
from discriminator import build_discriminator
from utils import preprocess_data, save_model, load_model
from wgan_gp import build_wgan, gradient_penalty, wasserstein_loss
from tensorflow.keras import backend as K

# Ruta para guardar el modelo generador
generator_model_path = 'generator_model.keras'

# Preprocesamiento de los datos
def process_chunk(chunk):
    if 'total_len' not in chunk.columns or 'protocol_m' not in chunk.columns or 'payload' not in chunk.columns:
        raise KeyError("Las columnas 'total_len', 'protocol_m', o 'payload' no existen en el DataFrame")
    
    # Seleccionar las columnas relevantes
    chunk = chunk[['total_len', 'protocol_m', 'payload']].copy()
    
    # Convertir 'total_len' a numérico
    chunk['total_len'] = pd.to_numeric(chunk['total_len'], errors='coerce')
    chunk['total_len'] = chunk['total_len'].fillna(chunk['total_len'].mean())
    
    # Convertir 'protocol_m' a columnas dummy (one-hot encoding)
    protocol_dummies = pd.get_dummies(chunk['protocol_m'], prefix='protocol')

    # Asegurarse de que todas las posibles columnas dummy estén presentes
    for col in ['protocol_TCP', 'protocol_UDP', 'protocol_Other']:
        if col not in protocol_dummies.columns:
            protocol_dummies[col] = 0  # Añadir la columna si falta

    # Concatenar las dummies con los datos originales
    chunk = pd.concat([chunk, protocol_dummies], axis=1)
    
    # Eliminar la columna original de 'protocol_m'
    chunk.drop(['protocol_m'], axis=1, inplace=True)
    
    # Convertir 'payload' a un valor numérico adecuado
    chunk['payload'] = chunk['payload'].apply(lambda x: int(x[:10], 16) if pd.notna(x) else 0)  # Truncar el payload a 10 caracteres

    return chunk


# Asegurarse de que todos los datos son numéricos
def ensure_numeric(data):
    return np.where(data == 'False', 0, np.where(data == 'True', 1, data))

# Cargar el dataset y procesarlo
def prepare_data(file_path):
    scaled_data = preprocess_data(file_path, process_chunk)

    # Asegurar que los datos del dataset son numéricos
    scaled_data = ensure_numeric(scaled_data).astype(np.float32)

    # Escalar total_len entre 0 y 1
    scaler = MinMaxScaler()
    scaled_data[:, 0] = scaler.fit_transform(scaled_data[:, 0].reshape(-1, 1)).reshape(-1)  # Asumimos que 'total_len' está en la primera columna
    return scaled_data, scaler

# Función para guardar el modelo generador
def save_generator(generator, path):
    generator.save(path)
    # Guardar scaler y latent_dim en un archivo .npy
    np.save('model_params.npy', {'scaler': scaler, 'latent_dim': latent_dim})
    print(f"Modelo generador guardado en {path}")

# Función para entrenar WGAN
def train_wgan(generator, discriminator, data, epochs, batch_size, latent_dim, critic_iterations=5, gp_weight=10):
    half_batch = batch_size // 2
    cpu_usage = []
    memory_usage = []
    epoch_times = []
    # Transformar los datos a DataFrame una vez fuera del bucle
    # Seleccionar solo las primeras 5 columnas de 'data' para que coincidan con column_names
    column_names = ['total_len', 'protocol_TCP', 'protocol_UDP', 'protocol_Other', 'payload']
    data = data[:, :5]  # Aquí seleccionamos solo las primeras 5 columnas
    real_data = pd.DataFrame(data, columns=column_names).astype(np.float32)
    
    # Optimizador para el discriminador
    d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)

    for epoch in range(epochs):
        start_time = time.time()

        for _ in range(critic_iterations):
            idx = np.random.randint(0, real_data.shape[0], half_batch)
            real_data_batch = real_data.iloc[idx].values  # Obtener un batch de los datos reales
            noise = np.random.normal(0, 1, (half_batch, latent_dim)).astype(np.float32)
            fake_data = generator.predict(noise)  # Asegúrate de pasar training=True

            # Calculamos la pérdida con GradientTape
            with tf.GradientTape() as tape:
                d_loss_real = discriminator(real_data_batch, training=True)  # Asegúrate de pasar training=True
                d_loss_fake = discriminator(fake_data, training=True)
                
                # Pérdida de Wasserstein
                wasserstein_loss_real = -tf.reduce_mean(d_loss_real)
                wasserstein_loss_fake = tf.reduce_mean(d_loss_fake)
                
                # Calcular el gradient penalty
                gp = gp_weight * gradient_penalty(discriminator, real_data_batch, fake_data, half_batch)
                
                # Pérdida total del discriminador
                d_loss = wasserstein_loss_real + wasserstein_loss_fake + gp
            
            # Calcular los gradientes del discriminador
            gradients = tape.gradient(d_loss, discriminator.trainable_variables)
            
            # Actualizar el discriminador
            d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        # Entrenar el generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(discriminator(generator(noise, training=True), training=True))
        
        # Optimizar el generador
        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        wgan.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)

        # Imprimir avance de la época
        if epoch % 1 == 0:
            print(f"{epoch}/{epochs}  [Time: {epoch_time:.2f}s]")

    # Guardar el generador entrenado después de terminar el entrenamiento
    save_generator(generator, generator_model_path)


# Ejecutar el entrenamiento solo si este archivo es el principal
if __name__ == "__main__":

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  
    file_path = r".\\ACI-IoT-2023-Payload.csv"
    
    # Preprocesar los datos
    print(Fore.YELLOW + "Preprocesando los datos..." + Style.RESET_ALL)
    scaled_data, scaler = prepare_data(file_path)
    
    # Parámetros del modelo
    latent_dim = 100
    data_dim = scaled_data.shape[1]  # Número de características después del procesamiento
    learning_rate = 1e-5
    epochs = 100
    batch_size = 64

    # Construir el generador y el discriminador
    print(Fore.YELLOW + "Construyendo el generador y el discriminador..." + Style.RESET_ALL)
    generator = build_generator(latent_dim)
    discriminator = build_discriminator((data_dim,))
    discriminator.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate))

    # Construir el modelo WGAN
    wgan = build_wgan(generator, discriminator, latent_dim, learning_rate)

    # Entrenar el modelo WGAN
    print(Fore.YELLOW + "Entrenando el modelo WGAN-GP..." + Style.RESET_ALL)
    train_wgan(generator, discriminator, scaled_data, epochs, batch_size, latent_dim)
    print(Fore.GREEN + "Modelo entrenado y guardado con éxito." + Style.RESET_ALL)
