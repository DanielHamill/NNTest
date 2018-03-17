import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sys import platform

if platform == "win32":
    import win_unicode_console as wc
    wc.enable()

mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

digit = mnist.train.images[121]
digit = np.array(digit)
print(digit)
pixels = digit.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
