from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

def get_array_from_image(image_name):
    img = Image.open(image_name)
    img = img.resize((28, 28))
    img = img.convert(mode='L')
    img = np.array(img)
    print(img.shape)
    img = img/4
    img = img
    print(img.shape)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print(img.shape)
    return img