import glob
import imp
import importlib
from msilib.schema import File
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
import time
from IPython import display
from utility import get_array_from_image


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(test_images[0].shape)
train_images = train_images/250
test_images[0] = get_array_from_image('4.png')
test_labels[0] = 4
print(type(train_images[1]))
print(train_labels[2])
plt.figure()
#Sets display to an image
plt.imshow(train_images[2])
#Adds colorbar to side
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure()
#Sets display to an image
plt.imshow(test_images[0])
#Adds colorbar to side
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    #Creates 25 subplots (5, 5) i = index (+1 because index is 1-25, not 0-24)
    plt.subplot(5,5,i+1)
    #Adds points on the x-y axis
    #Example: Sets adds 0, 0.5, and 1 ticks to graph: plt.xticks([0, 0.5, 1])
    #Example: Sets adds 0, 0.5, and 1 ticks to graph: plt.yticks([0, 0.5, 1])
    plt.xticks([]) #No ticks
    plt.yticks([]) #No ticks
    plt.grid(False)
    #Shows images
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #Lables the images.
    plt.xlabel(train_labels[i])
plt.show()
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
 
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100*np.max(predictions_array), true_label), color=color)
#Plots 
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()