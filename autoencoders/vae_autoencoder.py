import tensorflow as tf
from keras import layers, models
from keras import backend as K
import numpy as np

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape=(28, 28, 1), latent_dim=2):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    shape_before_flatten = K.int_shape(x)[1:]
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = models.Model(inputs, [z_mean, z_log_var, z])

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    vae = models.Model(inputs, decoded)

    # Loss function (reconstruction loss + KL divergence)
    reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(inputs), K.flatten(decoded)) * input_shape[0] * input_shape[1]
    reconstruction_loss = K.mean(reconstruction_loss)
    kl_loss = - 0.5 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder
