import tensorflow as tf
import matplotlib.pyplot as plt
from mlp_autoencoder import build_mlp_autoencoder
from cnn_autoencoder import build_cnn_autoencoder
from vae_autoencoder import build_vae

# Load MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Train MLP AutoEncoder
mlp_autoencoder = build_mlp_autoencoder(input_shape=(784,))
mlp_autoencoder.fit(x_train.reshape(-1, 784), x_train.reshape(-1, 784), epochs=10, batch_size=256, validation_data=(x_test.reshape(-1, 784), x_test.reshape(-1, 784)))

# Train CNN AutoEncoder
cnn_autoencoder = build_cnn_autoencoder(input_shape=(28, 28, 1))
cnn_autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# Train VAE AutoEncoder
vae, encoder = build_vae(input_shape=(28, 28, 1))
vae.fit(x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# --- Evaluation Section ---

# MLP AutoEncoder Evaluation
encoded_imgs = mlp_autoencoder.predict(x_test.reshape(-1, 784))
decoded_imgs = mlp_autoencoder.predict(x_test.reshape(-1, 784))

# Plot original vs reconstructed images (MLP)
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# CNN AutoEncoder Evaluation
decoded_imgs_cnn = cnn_autoencoder.predict(x_test)

# Plot original vs reconstructed images (CNN)
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_cnn[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# VAE AutoEncoder Evaluation
decoded_imgs_vae = vae.predict(x_test)

# Plot original vs reconstructed images (VAE)
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_vae[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
