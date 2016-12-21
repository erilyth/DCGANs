import random
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from tqdm import tqdm

# Uses Theano based tensors of shape (channels, rows, cols), for details see https://keras.io/backend/
# Use the MNIST dataset

h, w = 28, 28

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 1, h, w).astype('float32')
train_data /= 255.0
print(train_data.shape[1:])


# Discriminator Model
discriminator = Sequential()
discriminator.add(Convolution2D(128, 5, 5, subsample=(2,2), border_mode='same', activation='relu', input_shape=(1,28,28)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Dense(2, activation='softmax'))
discriminator_optim = Adam(lr=1e-3)
# Apply categorical loss at the output of shape (_, 2)
discriminator.compile(loss='categorical_crossentropy', optimizer=discriminator_optim)
print(discriminator.summary())


# Generator Model
generator = Sequential()
generator.add(Dense(512*7*7, input_shape=(100,)))
generator.add(LeakyReLU(0.1))
generator.add(Reshape([512, 7, 7]))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
generator.add(LeakyReLU(0.1))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
generator.add(LeakyReLU(0.1))
generator.add(Convolution2D(1, 1, 1, border_mode='same', activation='relu'))
generator.add(LeakyReLU(0.1))
generator_optim = Adam(lr=1e-4)
# Apply log loss at the output of shape (_, 1, 28, 28)
generator.compile(loss='binary_crossentropy', optimizer=generator_optim)
print(generator.summary())



