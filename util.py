from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

def get_array_from_image(image_name):
    img = Image.open(image_name)
    img = img.resize((28, 28))
    img = np.array(img)
    img = img/4
    return img