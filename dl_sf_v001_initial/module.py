# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def get_label_from_path(path):
	return int(path.split('\\')[-2])


def read_image(path):
	image = np.array(Image.open(path))

	# Reshaping for channel 1
	return image.reshape(image.shape[0], image.shape[1], 1)

def onehot_encode_label(path):
	onehot_label = unique_label_name == get_label_from_path
	onehot_label = onehot_label.astype(np.uint8)
	return onehot_label

def _read_py_function(path, label):
	image = read_image(path)
	label = np.array(label, dtype=np.uint8)
	return image.astype(np.int32), label

def _resize_function(image_decoded, label):
	image_decoded.set_shape([None, None, None])
	image_resized = tf.image.resize_images(image_decoded, [])
	return image_resized, label



# onehot-encoding through label name
class_name = get_label_from_path(path)

# Declare Variables
epochs = 1000

x = tf.placeholder(tf.float32, [None, 100 * 170])
W = tf.Variable(tf.zeros([100 * 170, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Setting cross-entropy model
y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Training the model with Gradient Descent Method
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(epochs):
	batch_xs, batch_ys = 
	sess.run(train_step, feed_dict=(x: batch_xs, y_: batch_ys))

# Printing how much this model is correct / effective
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict=(x: , y_:)))



# (x_train, y_train), (x_test, y_test) = testdata.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = keras.Sequential([
# 	tf.keras.layers.flatten(),
# 	tf.keras.layers.Dense(512, activation=tf.nn.relu),
# 	tf.keras.layers.Dropout(0.2),
# 	tf.keras.layers.Dense(3, activation=tf.nn.softmax
# )Mmodel.compile(optimizer=tf.train.AdamOptimizer(),
# 	loss='sparse_categorical_crossentropy',
# 	metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)