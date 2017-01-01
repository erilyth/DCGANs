import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D, Activation, merge
from keras.optimizers import Adam, sgd
from keras.layers.advanced_activations import LeakyReLU
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# Uses Theano based tensors of shape (channels, rows, cols), for details see https://keras.io/backend/

"""
# Algorithm Specifics:

"""

discriminator_losses = []
generator_losses = []
display_update = 500 # Save the models and update outputs every 100 iterations
backup_update = 3000 # Store a backup of the models every 1000 iterations
load_models = 1


def normalize_data(data):
    # Data shape would be (_, 1, 28, 28)
    data -= 128.0
    data /= 128.0
    return data


def unnormalize_data(data):
    # Data shape would be (_, 1, 28, 28)
    # data = np.clip(data, 0.5, 1.0)
    data *= 128.0
    data += 128.0
    return data


"""
Gather the training data from the MNIST dataset
"""
h, w = 28, 28 # Generated image height and width
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 1, h, w).astype('float32')
train_data = normalize_data(train_data)
print(train_data.shape[1:])


"""
Discriminator Model
"""
input_d = Input(shape=(1,28,28))

discriminator_t = Sequential()
discriminator_t.add(Convolution2D(128, 5, 5, subsample=(2,2), border_mode='same', input_shape=(1,28,28), init='glorot_uniform'))
discriminator_t.add(Activation('tanh'))
discriminator_t.add(Dropout(0.2))
discriminator_t.add(Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', init='glorot_uniform'))
discriminator_t.add(Activation('tanh'))
discriminator_t.add(Dropout(0.2))
discriminator_t.add(Flatten())

discriminator_model = discriminator_t(input_d)

type_predict = Dense(2, activation='softmax', name='type_predict')(discriminator_model)
class_predict = Dense(10, activation='softmax', name='class_predict')(discriminator_model)

discriminator = Model(input=input_d, output=[type_predict, class_predict])
discriminator_optim = Adam(lr=0.00001)
discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=discriminator_optim)
print(discriminator.summary())


"""
Generator Model
"""
input_g = Input(shape=(100,))
input_class = Input(shape=(10,))

generator_class = Dense(100, activation='tanh', input_shape=(10,))(input_class)
merge_layer = merge([input_g, generator_class], mode='mul')

generator_t = Sequential()
generator_t.add(Dense(512*7*7, input_shape=(100,), init='glorot_uniform'))
generator_t.add(Activation('tanh'))
generator_t.add(Reshape([512, 7, 7]))
generator_t.add(UpSampling2D(size=(2, 2)))
generator_t.add(Convolution2D(256, 3, 3, border_mode='same', init='glorot_uniform'))
generator_t.add(Activation('tanh'))
generator_t.add(UpSampling2D(size=(2, 2)))
generator_t.add(Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform'))
generator_t.add(Activation('tanh'))
generator_t.add(Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform'))
generator_t.add(Activation('tanh'))

generated_img = generator_t(merge_layer)

generator = Model(input=[input_g, input_class], output=generated_img)
generator_optim = Adam(lr=0.00001)
generator.compile(loss='binary_crossentropy', optimizer=generator_optim)
print(generator.summary())


"""
The final GAN architecture
"""
latent_noise = Input(shape=(100,))
image_class = Input(shape=(10,))
generated_image = generator([latent_noise, image_class])
print generated_image.shape
type_predicted, class_predicted = discriminator(generated_image)

GAN = Model(input=[latent_noise, image_class], output=[type_predicted, class_predicted])

GAN_optim = Adam(lr=0.00001)
GAN.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=GAN_optim)
GAN.summary()


if load_models == 1:
    print("Loading models from saved files!")
    discriminator.load_weights("discriminator.keras")
    generator.load_weights("generator.keras")


def toggle_trainable(network, state):
    network.trainable = state
    for layer in network.layers:
        layer.trainable = state


def sample_generation(class_id):
    class_vector = np.zeros([9, 10])
    class_vector[:,class_id] = 1.0
    sample_noise = np.random.uniform(-1.0, 1.0, size=[9, 100])
    generated_images = generator.predict([sample_noise, class_vector])
    generated_images = unnormalize_data(generated_images)
    for image_idx in range(len(generated_images)):
        plt.subplot(3, 3, image_idx+1)
        generated_image = generated_images[image_idx][0]
        plt.imshow(generated_image, cmap='gray')
    plt.show(block=False)
    time.sleep(3)
    plt.close('all')


def train_gan():
    for time_step in tqdm(range(30000)):
        if time_step % display_update == 0:
            # Display 9 randomly generated samples every display_update'th iteration
            class_id = np.random.randint(0,10)
            print(class_id)
            sample_generation(class_id)
            # Save the current models as well
            discriminator.save_weights("discriminator.keras", overwrite=True)
            generator.save_weights("generator.keras", overwrite=True)
        if time_step % backup_update == 0:
            discriminator.save_weights("discriminator_backup-" + str(time_step) + ".keras", overwrite=True)
            generator.save_weights("generator_backup-" + str(time_step) + ".keras", overwrite=True)

        """
        Generate some random noise and some random labels to use as inputs
        """
        batch_size = 64
        random_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        random_labels_idx = np.random.randint(0, 10, batch_size)
        random_labels = np.zeros([batch_size, 10])
        for idx in range(batch_size):
            random_labels[idx][random_labels_idx[idx]] = 1.0

        """
        To train the discriminator we need real images and generated images
        """
        # We first gather batch_size number of read images from the MNIST dataset
        train_idx = np.random.randint(0, len(train_data), size=batch_size)
        discrim_current_train = train_data[train_idx,:,:,:]
        discrim_current_train_labels_idx = train_labels[train_idx]
        discrim_current_train_labels = np.zeros([batch_size, 10])
        for idx in range(batch_size):
            discrim_current_train_labels[idx][discrim_current_train_labels_idx[idx]] = 1.0

        # We now use the random noise and random labels to generate fake images
        discrim_current_noise = random_noise
        discrim_current_noise_labels = random_labels
        discrim_generated_train = generator.predict([discrim_current_noise, discrim_current_noise_labels])

        # Concatenate both the real and the fake data to produce data for an epoch of training on the discriminator
        discrim_current_train = np.concatenate((discrim_current_train, discrim_generated_train))
        discrim_current_train_labels = np.concatenate((discrim_current_train_labels, discrim_current_noise_labels))
        discrim_current_train_type = np.zeros(shape=[batch_size * 2, 2])
        # The first half of the samples are real data whereas the second half are generated
        for ix in range(batch_size):
            discrim_current_train_type[ix, 1] = np.random.uniform(0.7, 1.2)
            discrim_current_train_type[ix, 0] = np.random.uniform(0.0, 0.3)
            discrim_current_train_type[batch_size+ix, 0] = np.random.uniform(0.7, 1.2)
            discrim_current_train_type[batch_size + ix, 1] = np.random.uniform(0.0, 0.3)

        discriminator_loss_cur = discriminator.train_on_batch(discrim_current_train, [discrim_current_train_type, discrim_current_train_labels])
        discriminator_losses.append(discriminator_loss_cur)

        """
        To train the generator use the random noise and random labels from earlier
        """
        gen_current_train = random_noise
        gen_current_train_labels = random_labels
        gen_current_train_type = np.zeros(shape=[batch_size, 2])
        # When we train the generator we want it to fool the discriminator so we use the opposite labels
        # We use gen_current_labels[:, 1] = 1 instead of using gen_current_labels[:, 0] = 1
        for ix in range(batch_size):
            gen_current_train_type[:, 1] = np.random.uniform(0.7, 1.2)
            gen_current_train_type[:, 0] = np.random.uniform(0.0, 0.3)
        generator_loss_cur = GAN.train_on_batch([gen_current_train, gen_current_train_labels], [gen_current_train_type, gen_current_train_labels])
        generator_losses.append(generator_loss_cur)


        print("Time Step: ", time_step, ", Discriminator Loss: ", discriminator_loss_cur, ", Generator Loss: ", generator_loss_cur)


train_gan()
# for x in range(1000):
#    sample_generation(np.random.randint(0,10))

plt.show()
