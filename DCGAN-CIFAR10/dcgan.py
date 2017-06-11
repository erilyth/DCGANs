import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D, Activation, AveragePooling2D
from keras.optimizers import Adam, sgd
from keras.layers.advanced_activations import LeakyReLU
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# Uses Theano based tensors of shape (channels, rows, cols), for details see https://keras.io/backend/

"""
# Algorithm Specifics:
1) Uses the CIFAR10 dataset
2) The Generator takes in a random noise vector of shape (,100) as input and generates an output of shape (,1,28,28)
3) The Discriminator takes in an input image of shape (,1,28,28) and gives an output of shape (,2)
4) The outputs of the discriminator are probabilities. In the output labels, 0 corresponds to generated and 1 corresponds to real.  
5) When we train the discriminator we use the correct output labels but when we train the generator we invert the labels, this ensures that the discriminator tries to maximize its prediction accuracy whereas the generator tries to minimize it.
"""

discriminator_losses = []
generator_losses = []
display_update = 500 # Save the models and update outputs every 100 iterations
backup_update = 5000 # Store a backup of the models every 1000 iterations
load_models = 0


def normalize_data(data):
    # Data shape would be (_, 1, 32, 32)
    data /= 256.0
    data -= 0.5
    return data


def unnormalize_data(data):
    # Data shape would be (_, 1, 32, 32)
    # data = np.clip(data, 0.5, 1.0)
    data += 0.5
    data *= 256.0
    return data

h, w = 32, 32
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 3, h, w).astype('float32')
train_data = normalize_data(train_data)
print(train_data.shape[1:])

# Discriminator Model
discriminator = Sequential()
discriminator.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(3,32,32)))
discriminator.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(128, 5, 5, border_mode='same'))
discriminator.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(256, 5, 5, border_mode='same'))
discriminator.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(512, 5, 5, border_mode='same'))
discriminator.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Convolution2D(1, 5, 5, border_mode='same'))
discriminator.add(AveragePooling2D(pool_size=(2, 2), border_mode='valid'))
discriminator.add(Flatten())
discriminator.add(Activation('tanh'))
discriminator_optim = SGD(lr=0.01, momentum=0.9, nesterov=True)
# Apply categorical loss at the output of shape (_, 1)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optim)
print(discriminator.summary())

# Generator Model
generator = Sequential()
generator.add(Dense(512*2*2, input_shape=(100,)))
generator.add(BatchNormalization(mode=0))
generator.add(Reshape([512, 2, 2]))
generator.add(Convolution2D(512, 5, 5, border_mode='same'))
generator.add(BatchNormalization(mode=0, axis=1))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(256, 5, 5, border_mode='same'))
generator.add(BatchNormalization(mode=0, axis=1))
generator.add(LeakyReLU(0.2))
generator.add(Dropout(0.4))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(128, 5, 5, border_mode='same'))
generator.add(BatchNormalization(mode=0, axis=1))
generator.add(LeakyReLU(0.2))
generator.add(Dropout(0.4))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(64, 5, 5, border_mode='same'))
generator.add(BatchNormalization(mode=0, axis=1))
generator.add(LeakyReLU(0.2))
generator.add(Dropout(0.4))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(3, 5, 5, border_mode='same'))
generator.add(Activation('tanh'))
generator_optim = Adam(lr=1e-4)
# Apply log loss at the output of shape (_, 3, 32, 32)
generator.compile(loss='binary_crossentropy', optimizer=generator_optim)
print(generator.summary())

# The final GAN architecture
GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)
GAN_optim = Adam(lr=1e-4)
GAN.compile(loss='binary_crossentropy', optimizer=GAN_optim)
GAN.summary()

if load_models == 1:
    print("Loading models from saved files!")
    discriminator.load_weights("discriminator.keras")
    generator.load_weights("generator.keras")


def toggle_trainable(network, state):
    network.trainable = state
    for layer in network.layers:
        layer.trainable = state


def sample_generation(iter_num):
    global generator
    sample_noise = np.random.normal(loc=0.0, scale=1.0, size=[9, 100])
    generated_images = generator.predict(sample_noise)
    generated_images = unnormalize_data(generated_images)
    for image_idx in range(len(generated_images)):
        plt.subplot(3, 3, image_idx+1)
        generated_image = generated_images[image_idx].transpose(1,2,0)
        plt.imshow(generated_image)
    #plt.show(block=False)
    plt.savefig('sample_'+str(iter_num)+'.png')
    #time.sleep(3)
    #plt.close('all')


def pretrain_discriminator():
    global generator
    global discriminator
    # Call this before training the GAN to start with a trained discriminator
    current_train = train_data[:, :, :, :]
    current_noise = np.random.uniform(-1.0, 1.0, size=[len(train_data), 100])
    current_train = np.concatenate((current_train, generator.predict(current_noise)))
    current_labels = np.zeros(shape=[len(current_train)])
    # The first half of the samples are real data whereas the second half are generated
    current_labels[:int(len(current_train)/2)] = 1.0
    current_labels[int(len(current_train)/2):] = 0.0
    print("Starting to pre-train the discriminator!")
    discriminator.fit(current_train, current_labels, nb_epoch=1, batch_size=32)
    # Save the trained discriminator weights
    discriminator.save_weights("discriminator.keras", overwrite=True)
    generator.save_weights("generator.keras", overwrite=True)

def train_gan():
    global generator
    global discriminator
    global generator_losses
    global discriminator_losses
    global GAN
    for time_step in tqdm(range(100000)):
        if time_step % display_update == 0:
            # Display 9 randomly generated samples every display_update'th iteration
            sample_generation(time_step)
            # Save the current models as well
            discriminator.save_weights("discriminator.keras", overwrite=True)
            generator.save_weights("generator.keras", overwrite=True)
        if time_step % backup_update == 0:
            discriminator.save_weights("discriminator_backup-" + str(time_step) + ".keras", overwrite=True)
            generator.save_weights("generator_backup-" + str(time_step) + ".keras", overwrite=True)

        batch_size = 4
        random_noise = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, 100])

        train_idx = np.random.randint(0, len(train_data), size=batch_size)
        discrim_current_orig_train = train_data[train_idx,:,:,:].copy()
        discrim_current_noise = random_noise.copy()
        discrim_generated_train = generator.predict(discrim_current_noise)

        # print("Starting to train the discriminator!")
	discrim_current_train = discrim_current_orig_train.copy()
	discrim_current_labels = np.zeros(shape=[batch_size])
	# Original data samples have a label 1
	for ix in range(batch_size):
	    if np.random.uniform(0.0, 1.0) < 0.95:
	        discrim_current_labels[ix] = np.random.uniform(0.7, 1.2)
	    else:
		discrim_current_labels[ix] = np.random.uniform(0.0, 0.3)
	discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, discrim_current_labels)

        discrim_current_train = discrim_generated_train.copy()
        discrim_current_labels = np.zeros(shape=[batch_size])
	# Generated samples have a label 0
	for ix in range(batch_size):
	    if np.random.uniform(0.0, 1.0) < 0.95:
	        discrim_current_labels[ix] = np.random.uniform(0.0, 0.3)
	    else:
		discrim_current_labels[ix] = np.random.uniform(0.7, 1.2)
        discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, discrim_current_labels)
        discriminator_losses.append(discriminator_loss_cur)

	toggle_trainable(discriminator, False)
        gen_current_train = random_noise.copy()
        gen_current_labels = np.zeros(shape=[batch_size])
        # When we train the generator we want it to fool the discriminator so we use the opposite labels
        # We use gen_current_labels[:] = 1 instead of using gen_current_labels[:] = 0
        for ix in range(batch_size):
            gen_current_labels[ix] = np.random.uniform(0.7, 1.2)
        #print("Starting to train the generator!")
        generator_loss_cur = GAN.train_on_batch(gen_current_train, gen_current_labels)
        generator_losses.append(generator_loss_cur)
	toggle_trainable(discriminator, True)

        print("Time Step: ", time_step, ", Discriminator Loss: ", discriminator_loss_cur, ", Generator Loss: ", generator_loss_cur)

# pretrain_discriminator()
train_gan()

for x in range(1000):
    sample_generation()

plt.show()
