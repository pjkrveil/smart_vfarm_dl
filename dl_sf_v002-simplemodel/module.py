# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib


class sfModel(keras.Model):
	# construct three layers

  def __init__(self, num_classes=3):
    super(sfModel, self).__init__(name='sf_detector')
    self.num_classes = num_classes

    # Define your layers here.
    self.flatten_1 = keras.layers.Flatten(input_shape=(130, 130, 3))

    self.dense_1 = keras.layers.Dense(4225, activation='tf.nn.relu')
    self.dropout_1 = keras.layers.Dropout(0.2)

    self.dense_2 = keras.layers.Dense(676, activation='tf.nn.relu')
    self.dropout_2 = keras.layers.Dropout(0.2)

    self.dense_3 = keras.layers.Dense(169, activation='tf.nn.relu')
    self.dropout_3 = keras.layers.Dropout(0.2)

    self.dense

    self.dense_2 = keras.layers.Dense(num_classes, activation='tf.nn.softmax')

    def create_model(self):
    	model = Sequential()

    	# Block 1
    	model.add(InputLayer(input_sahpe=(130, 130, 3)))
    	model.add(Conv2D(4, (3 , 3), activation='relu', padding='same'))
    	model.add(Conv2D(4, (3 , 3), activation='relu', padding='same'))
    	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    	# Block 2
    	model.add(Conv2D(25, (3 , 3), activation='relu', padding='same'))
    	model.add(Conv2D(25, (3 , 3), activation='relu', padding='same'))
    	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    	# Block 3
    	model.add(Conv2D(100, (3 , 3), activation='relu', padding='same'))
    	model.add(Conv2D(100, (3 , 3), activation='relu', padding='same'))
    	model.add(Conv2D(100, (3 , 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))






  def call_model(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


# Instantiates the subclassed model.
model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)