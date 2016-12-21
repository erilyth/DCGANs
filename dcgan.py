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

discriminator_losses = []
generator_losses = []


def normalize_data(data):
    # Data shape would be (_, 1, 28, 28)
    data -= 128.0
    data /= 255.0
    return data


def unnormalize_data(data):
    # Data shape would be (_, 1, 28, 28)
    data *= 255.0
    data += 128.0
    return data


def evaluate_accuracy(labels1, labels2):
    assert len(labels1) == len(labels2)
    score = 0.0
    for idx in range(len(labels1)):
        if np.argmax(labels1[idx]) == np.argmax(labels2[idx]):
            score += 1.0
    return score*100.0/len(labels1)


h, w = 28, 28
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 1, h, w).astype('float32')
train_data = normalize_data(train_data)
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

# The final GAN architecture
GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)
GAN_optim = Adam(lr=1e-4)
GAN.compile(loss='categorical_crossentropy', optimizer=GAN_optim)
GAN.summary()


def pretrain_discriminator():
    # Call this before training the GAN to start with a trained discriminator
    current_train = train_data[:, :, :, :]
    current_noise = np.random.uniform(0, 1, size=[len(train_data), 100])
    current_train = np.concatenate((current_train, generator.predict(current_noise)))
    current_labels = np.zeros(shape=[len(current_train), 2])
    # The first half of the samples are real data whereas the second half are generated
    current_labels[:int(len(current_train)/2),1] = 1
    current_labels[int(len(current_train)/2):,0] = 1
    discriminator.fit(current_train, current_labels, nb_epoch=1, batch_size=64)


def train_gan():
    for time_step in tqdm(range(10000)):
        batch_size = 64
        random_noise = np.random.uniform(0, 1, size=[batch_size, 100])

        train_idx = np.random.randint(0, len(train_data), size=batch_size)
        discrim_current_train = train_data[train_idx,:,:,:]
        discrim_current_noise = random_noise
        discrim_generated_train = generator.predict(discrim_current_noise)

        discrim_current_train = np.concatenate((discrim_current_train, discrim_generated_train))
        discrim_current_labels = np.zeros(shape=[batch_size * 2, 2])
        # The first half of the samples are real data whereas the second half are generated
        discrim_current_labels[:batch_size, 1] = 1
        discrim_current_labels[batch_size:, 0] = 1

        discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, discrim_current_labels)
        discriminator_losses.append(discriminator_loss_cur)

        gen_current_train = random_noise
        gen_current_labels = np.zeros(shape=[batch_size, 2])
        # When we train the generator we want it to fool the discriminator so we use the opposite labels
        # We use gen_current_labels[:, 0] = 1 instead of using gen_current_labels[:, 1] = 1
        gen_current_labels[:, 0] = 1
        generator_loss_cur = GAN.train_on_batch(gen_current_train, gen_current_labels)
        generator_losses.append(generator_loss_cur)

        print("Time Step: ", time_step, ", Discriminator Loss: ", discriminator_loss_cur, ", Generator Loss: ", generator_loss_cur)
