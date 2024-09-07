import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_data, fake_data, batch_size):
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity_interpolated = discriminator(interpolated)
    gradients = tape.gradient(validity_interpolated, [interpolated])[0]
    gradients_sqr = tf.square(gradients)
    gradients_l2_norm = tf.sqrt(tf.reduce_sum(gradients_sqr, axis=np.arange(1, gradients_sqr.shape.ndims)))
    gradient_penalty = tf.reduce_mean((gradients_l2_norm - 1.0) ** 2)
    return gradient_penalty


def build_wgan(generator, discriminator, latent_dim, learning_rate):
    # Input del generador es el vector latente
    z = Input(shape=(latent_dim,))
    
    # La salida del generador son 4 características
    data = generator(z)
    
    # Pasamos la salida del generador (de tamaño 4) al discriminador
    valid = discriminator(data)
    
    # Definir el modelo completo WGAN
    wgan = Model(z, valid)
    
    # Compilar usando la función de pérdida de Wasserstein
    wgan.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate))

    return wgan