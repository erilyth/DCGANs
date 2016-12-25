import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D, Activation
from keras.optimizers import Adam, sgd
from keras.layers.advanced_activations import LeakyReLU
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# Uses Theano based tensors of shape (channels, rows, cols), for details see https://keras.io/backend/
# Use the MNIST dataset

discriminator_losses = []
generator_losses = []
display_update = 10 # Save the models and update outputs every 5 iterations
backup_update = 400 # Store a backup of the models every 400 iterations
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
discriminator.add(Convolution2D(128, 5, 5, subsample=(2,2), border_mode='same', input_shape=(1,28,28), init='glorot_uniform'))
discriminator.add(Activation('tanh'))
discriminator.add(Dropout(0.2))
discriminator.add(Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', init='glorot_uniform'))
discriminator.add(Activation('tanh'))
discriminator.add(Dropout(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(512, init='glorot_uniform'))
discriminator.add(Activation('tanh'))
discriminator.add(Dropout(0.2))
discriminator.add(Dense(2, activation='softmax', init='glorot_uniform'))
discriminator_optim = sgd(lr=0.01)
# Apply categorical loss at the output of shape (_, 2)
discriminator.compile(loss='categorical_crossentropy', optimizer=discriminator_optim)
print(discriminator.summary())

# Generator Model
generator = Sequential()
generator.add(Dense(512*7*7, input_shape=(100,), init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.add(Reshape([512, 7, 7]))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(256, 3, 3, border_mode='same', init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.add(Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform'))
generator.add(Activation('tanh'))
generator_optim = Adam(lr=0.00001)
# Apply log loss at the output of shape (_, 1, 28, 28)
generator.compile(loss='binary_crossentropy', optimizer=generator_optim)
print(generator.summary())

# The final GAN architecture
GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)
GAN_optim = Adam(lr=0.00001)
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
    sample_noise = np.random.uniform(-0.5, 0.5, size=[9, 100])
    generated_images = generator.predict(sample_noise)
    generated_images = unnormalize_data(generated_images)
    for image_idx in range(len(generated_images)):
        plt.subplot(3, 3, image_idx+1)
        generated_image = generated_images[image_idx][0]
        plt.imshow(generated_image, cmap='gray')
    plt.show(block=False)
    time.sleep(3)
    plt.close('all')


def pretrain_discriminator():
    global generator
    global discriminator
    # Call this before training the GAN to start with a trained discriminator
    current_train = train_data[:, :, :, :]
    current_noise = np.random.uniform(-0.5, 0.5, size=[len(train_data), 100])
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
    global generator
    global discriminator
    global generator_losses
    global discriminator_losses
    for time_step in tqdm(range(100000)):
        if time_step % display_update == 0:
            # Display 9 randomly generated samples every display_update'th iteration
            sample_generation()
            # Save the current models as well
            discriminator.save_weights("discriminator.keras")
            generator.save_weights("generator.keras")
        if time_step % backup_update == 0:
            discriminator.save_weights("discriminator_backup-" + str(time_step) + ".keras")
            generator.save_weights("generator_backup-" + str(time_step) + ".keras")

        batch_size = 64
        random_noise = np.random.uniform(-0.5, 0.5, size=[batch_size, 100])

        # toggle_trainable(discriminator, True)
        # toggle_trainable(generator, False)
        train_idx = np.random.randint(0, len(train_data), size=batch_size)
        discrim_current_train = train_data[train_idx,:,:,:]
        discrim_current_noise = random_noise
        discrim_generated_train = generator.predict(discrim_current_noise)

        discrim_current_train = np.concatenate((discrim_current_train, discrim_generated_train))
        discrim_current_labels = np.zeros(shape=[batch_size * 2, 2])
        # The first half of the samples are real data whereas the second half are generated
        for ix in range(batch_size):
            discrim_current_labels[ix, 1] = np.random.uniform(0.7, 1.2)
            discrim_current_labels[ix, 0] = np.random.uniform(0.0, 0.3)
            discrim_current_labels[batch_size+ix, 0] = np.random.uniform(0.7, 1.2)
            discrim_current_labels[batch_size + ix, 1] = np.random.uniform(0.0, 0.3)

        # print("Starting to train the discriminator!")
        discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, discrim_current_labels)
        discriminator_losses.append(discriminator_loss_cur)
        # toggle_trainable(generator, True)
        # toggle_trainable(discriminator, False)

        # toggle_trainable(generator, True)
        # toggle_trainable(discriminator, False)
        gen_current_train = random_noise
        gen_current_labels = np.zeros(shape=[batch_size, 2])
        # When we train the generator we want it to fool the discriminator so we use the opposite labels
        # We use gen_current_labels[:, 0] = 1 instead of using gen_current_labels[:, 1] = 1
        for ix in range(batch_size):
            gen_current_labels[:, 1] = np.random.uniform(0.7, 1.2)
            gen_current_labels[:, 0] = np.random.uniform(0.0, 0.3)
        # print("Starting to train the generator!")
        generator_loss_cur = GAN.train_on_batch(gen_current_train, gen_current_labels)
        generator_losses.append(generator_loss_cur)
        # toggle_trainable(discriminator, True)
        # toggle_trainable(generator, False)

        print("Time Step: ", time_step, ", Discriminator Loss: ", discriminator_loss_cur, ", Generator Loss: ", generator_loss_cur)

# pretrain_discriminator()
train_gan()
plt.show()