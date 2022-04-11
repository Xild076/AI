from AI import AI
import tensorflow as tf
import util

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

ai = AI((28, 28))
ai.train(train_images, train_labels, 10)
ai.test(test_images, test_labels)
info = util.get_array_from_image('4.png')
print(info.shape)
info_list = [info]
print(info_list)
prediction = ai.predict(info_list)
ai.show_image(0, prediction, 4, info_list)