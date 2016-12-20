import random
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use the MNIST dataset
h, w = 28, 28

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_size = train_data.shape[0]
# Batch inputs to keras models require this shape
train_data = train_data.reshape(train_size, 1, h, w).astype('float32')
train_data /= 255.0

