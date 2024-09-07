import tensorflow as tf
from tensorflow.keras import layers, models

def build_discriminator(input_shape):
    model = models.Sequential()
    
    # Cambiar el input_shape a (5,) para que coincida con la salida del generador
    model.add(layers.Dense(1024, input_shape=(5,)))  # Cambiado a 5 dimensiones
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))
    
    # La salida del discriminador es una probabilidad real
    model.add(layers.Dense(1))
    
    return model

