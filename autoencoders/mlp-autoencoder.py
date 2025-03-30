import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp_autoencoder(input_shape=(784,)):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu')(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(x)

    # Define model
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder
