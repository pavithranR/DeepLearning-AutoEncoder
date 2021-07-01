import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from numpy import random

#Fashion mnist data set import and split for training and testing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#data type conversion
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#normalizing the data
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

latent_dim = 64 
svmIn = 0
#Auto encoder 
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        #encoder layers 128, 64, 32
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(latent_dim, activation='relu'),
          layers.Dense(32, activation='relu', name="forsvm"),
        ])
        #decoder layers 64, 128, 784
        self.decoder = tf.keras.Sequential([
          layers.Dense(latent_dim, activation='relu'),
          layers.Dense(128, activation='relu'),
          layers.Dense(784, activation='relu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(metrics='accuracy',optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
print("Epoch and loss plot is commented in code")
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

x_test = x_test.reshape((len(x_test), 28,28))
decoded_imgs = decoded_imgs.reshape((len(decoded_imgs),28,28))
#random number array for choosing image
randImgIndex = random.choice(10000, size=3, replace=False)
n = 2
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[randImgIndex[i]])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[randImgIndex[i]])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#*******************************************************For plotting the loss*************************************************
#*****************************************************************************************************************************
#This code chunck works well in anaconda and google colab but fails in local computer. So included the plot in report
history = autoencoder
def plot_his(history):
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(history.epoch, np.array(history.history['loss']), label = 'Train loss -training data')
    plt.plot(history.epoch, np.array(history.history['val_loss']),label ='val loss -cross validation data')
    plt.legend()
    plt.ylim([0.015,0.057])

plot_his(history)    
#*****************************************************************************************************************************

#experimenting with more layer and width
print("----------------------------------------------------------------------------------")
print("----------------Fully connected autoencoder with added layers---------------------")
#Fashion mnist data set import and split for training and testing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#data type conversion
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#normalizing the data
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

class DeepAutoEncoder(Model):
    def __init__(self, latent_dim):
        super(DeepAutoEncoder, self).__init__()
        self.latent_dim = latent_dim   
        #encoder layers 128, 64, 32
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(256, activation='relu'),
          layers.Dense(128, activation='relu'),
          layers.Dense(latent_dim, activation='relu'),
          layers.Dense(32, activation='relu', name="forsvm"),
        ])
        #decoder layers 64, 128, 784
        self.decoder = tf.keras.Sequential([
          layers.Dense(latent_dim, activation='relu'),
          layers.Dense(128, activation='relu'),
          layers.Dense(256, activation='relu'),
          layers.Dense(784, activation='relu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

deepAutoEncoder = DeepAutoEncoder(latent_dim)

deepAutoEncoder.compile(metrics='accuracy',optimizer='adam', loss=losses.MeanSquaredError())

deepAutoEncoder.fit(x_train, x_train, epochs=30, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

deepencoded_imgs = deepAutoEncoder.encoder(x_test).numpy()
deepdecoded_imgs = deepAutoEncoder.decoder(deepencoded_imgs).numpy()

x_test = x_test.reshape((len(x_test), 28,28))
deepdecoded_imgs = deepdecoded_imgs.reshape((len(deepdecoded_imgs),28,28))
#random number array for choosing image
drandImgIndex = random.choice(10000, size=3, replace=False)
n = 2
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[drandImgIndex[i]])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(deepdecoded_imgs[drandImgIndex[i]])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#*******************************************************For plotting the loss*************************************************
#*****************************************************************************************************************************
#This code chunck works well in anaconda and google colab but fails in local computer. So included the plot in report
history = deepAutoEncoder
def plot_his(history):
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("Mean squared error")
    plt.plot(history.epoch, np.array(history.history['loss']), label = 'Train loss -training data')
    plt.plot(history.epoch, np.array(history.history['val_loss']),label ='val loss -cross validation data')
    plt.legend()
    plt.ylim([0.015,0.057])

plot_his(history)
#*****************************************************************************************************************************
