import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class AI(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=self.input_size), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def train(self, train_image, train_label, epochs):
        self.model.fit(train_image, train_label, epochs=epochs)
    
    def test(self, test_image, test_label):
        test_loss, test_acc = self.model.evaluate(test_image,  test_label, verbose=2)
        print("Losses:", test_loss, " Accuracy:", test_acc)
    
    def predict(self, predict_image):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(predict_image)
        return predictions
    
    def plot_image(i, predictions_array, true_label, img):
        print(true_label)
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
    
    def show_image(i, prediction_array, true_label, img):
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        AI.plot_image(i, prediction_array[i], true_label, img)
        plt.subplot(1,2,2)
        AI.plot_value_array(i, prediction_array[i],  true_label)
        plt.show()