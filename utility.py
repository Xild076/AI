from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

def get_array_from_image(image_name):
    """img = Image.open(image_name)
    img = img.resize((28, 28))
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    img = tf.keras.utils.img_to_array(img)
    img.resize(28,28)
    print(img.shape)
    print(img)
    return img"""
    """img = Image.open(image_name)
    img_depixelized = img.resize((28, 28))
    img_standardized = tf.image.per_image_standardization(img_depixelized)
    print(type(img_standardized))
    print(img_standardized[-1])
    return img_standardized[-1]"""
    img = Image.open(image_name)
    img = img.resize((28, 28))
    img = 
    print(img.shape, "Shape")
    return img
    #return np.resize(imread('4.png'), (28, 28))