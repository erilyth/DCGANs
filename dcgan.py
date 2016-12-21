import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import keras
import matplotlib.pyplot as plt
from tqdm import tqdm


# Uses Theano based tensors of shape (channels, rows, cols), for details see https://keras.io/backend/
# Use the MNIST dataset

discriminator_losses = []
generator_losses = []
display_update = 5 # Save the models and update outputs every 5 iterations
backup_update = 15 # Store a backup of the models every 15 iterations
load_models = 1


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

if load_models == 1:
    print("Loading models from saved files!")
    discriminator.load_weights("discriminator.keras")
    generator.load_weights("generator.keras")


def toggle_trainable(network, state):
    network.trainable = state
    for layer in network.layers:
        layer.trainable = state


def sample_generation():
    sample_noise = np.random.uniform(0, 1, size=[9, 100])
    generated_images = generator.predict(sample_noise)
    generated_images = unnormalize_data(generated_images)
    for image_idx in range(len(generated_images)):
        plt.subplot(3, 3, image_idx+1)
        generated_image = generated_images[image_idx][0]
        plt.imshow(generated_image, cmap='gray')
    plt.show(block=False)


def pretrain_discriminator():
    # Call this before training the GAN to start with a trained discriminator
    current_train = train_data[:, :, :, :]
    current_noise = np.random.uniform(0, 1, size=[len(train_data), 100])
    current_train = np.concatenate((current_train, generator.predict(current_noise)))
    current_labels = np.zeros(shape=[len(current_train), 2])
    # The first half of the samples are real data whereas the second half are generated
    current_labels[:int(len(current_train)/2),1] = 1
    current_labels[int(len(current_train)/2):,0] = 1
    print("Starting to pre-train the discriminator!")
    discriminator.fit(current_train, current_labels, nb_epoch=1, batch_size=64)
    # Save the trained discriminator weights
    discriminator.save_weights("discriminator.keras")
    generator.save_weights("generator.keras")

def train_gan():
    for time_step in tqdm(range(100000)):
        if time_step % display_update == 0:
            # Display 9 randomly generated samples every display_update'th iteration
            sample_generation()
            # Save the current models as well
            discriminator.save_weights("discriminator.keras")
            generator.save_weights("generator.keras")
            if time_step % backup_update == 0:
                discriminator.save_weights("discriminator_backup.keras")
                generator.save_weights("generator_backup.keras")

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

        print("Starting to train the discriminator!")
        discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, discrim_current_labels)
        discriminator_losses.append(discriminator_loss_cur)

        toggle_trainable(discriminator, False)
        gen_current_train = random_noise
        gen_current_labels = np.zeros(shape=[batch_size, 2])
        # When we train the generator we want it to fool the discriminator so we use the opposite labels
        # We use gen_current_labels[:, 0] = 1 instead of using gen_current_labels[:, 1] = 1
        gen_current_labels[:, 0] = 1
        print("Starting to train the generator!")
        generator_loss_cur = GAN.train_on_batch(gen_current_train, gen_current_labels)
        generator_losses.append(generator_loss_cur)
        toggle_trainable(discriminator, True)

        print("Time Step: ", time_step, ", Discriminator Loss: ", discriminator_loss_cur, ", Generator Loss: ", generator_loss_cur)

if load_models == 0:
    pretrain_discriminator()
train_gan()