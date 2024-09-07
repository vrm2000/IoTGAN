import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    # Cambiar la salida a 5 características
    model.add(layers.Dense(5, activation='linear'))  # Ajustado a 5 características
    
    return model


