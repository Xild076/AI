import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

#Datasets
fashion_mnist = tf.keras.datasets.fashion_mnist
#Loading datasets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Sorting information
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Showing a figure
plt.figure()
#Sets display to an image
plt.imshow(train_images[0])
#Adds colorbar to side
plt.colorbar()
plt.grid(False)
plt.show()
#Sets color to black and white
train_images = train_images / 255.0
#0-1 is white-black gradient
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
#Creates a 10*10 viewing area
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
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#Sequential groups a linear stack of layers into a tf.keras.Model (Model groups layers into an object with training and inference features.). 
#Flatten means converting data to 1-D (Since image was 28*28)
#Dense is create fully connected layers, in which every output depends on every input, FIND OUT WHAT THAT MEANS.
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
#Compile = Adding settings. Optimizer = Adam optimizer setting. Losses = Measures how accurate AI is. Metrics: Measures percent images are accurately determined 
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#.fits starts training, with train_images and train_labels. Epochs = number of runs
model.fit(train_images, train_labels, epochs=10)
#model.evaluate tests with test_images, test_loss is faliure rates, test_acc is accuracy. Verbose is setting for evaluation.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc, '\nTest losses:', test_loss)
#Propability is use after testing. Softmax is used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#probability_model is finilazed with keras.Sequential. Predict is used to run.
predictions = probability_model.predict(test_images)
#predictions is results. Gives to 9 probabilities.
print(predictions[0])
#Pulls highest probability from list.
print(np.argmax(predictions[0]))
#Result that is should be.
print(test_labels[0])
#Plots results
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

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)
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