import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

hl_nodes_1 = 500
hl_nodes_2 = 500
hl_nodes_3 = 500

n_classes = 10
batch_size = 100

# placeholder variables for x and y (in and out)
x = tf.placeholder(tf.float32, shape=[0, 784])

tf
