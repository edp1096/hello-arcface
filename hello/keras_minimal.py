"""
https://www.tensorflow.org/guide/keras/sequential_model
https://www.tensorflow.org/guide/keras/train_and_evaluate
https://keras.io/examples/vision/mnist_convnet
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np


num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=keras.metrics.Accuracy(),
)


batch_size = input_shape[0] * input_shape[1]
epochs = 10

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
