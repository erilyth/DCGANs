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

"""
# Algorithm Specifics:
1) Uses the MNIST dataset
2) The Generator takes in a random noise vector of shape (,100) as input and generates an output of shape (,1,28,28)
3) The Discriminator takes in an input image of shape (,1,28,28) and gives an output of shape (,2)
4) The outputs of the discriminator are probabilities. In the output labels, 0 corresponds to generated and 1 corresponds to real.  
5) When we train the discriminator we use the correct output labels but when we train the generator we invert the labels, this ensures that the discriminator tries to maximize its prediction accuracy whereas the generator tries to minimize it.
"""

discriminator_losses = []
generator_losses = []
display_update = 10 # Save the models and update outputs every 100 iterations
backup_update = 1000 # Store a backup of the models every 1000 iterations
load_models = 0


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


h, w = 28, 28 # Generated image height and width
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 1, h, w).astype('float32')
train_data = normalize_data(train_data)
print(train_data.shape[1:])

# Discriminator Model
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

# Generator Model
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

# The final GAN architecture
latent_noise = Input(shape=(100,))
image_class = Input(shape=(10,))
generated_image = generator([latent_noise, image_class])
type_predict, class_predict = discriminator(generated_image)

GAN = Model(input=[latent_noise, image_class], output=[type_predict, class_predict])

GAN_optim = Adam(lr=0.00001)
GAN.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=GAN_optim)
GAN.summary()

if load_models == 1:
    print("Loading models from saved files!")
    discriminator.load_weights("discriminator.keras")
    generator.load_weights("generator.keras")


