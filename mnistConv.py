#https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)
from sys import platform

if platform == "win32":
    import win_unicode_console as wc
    wc.enable()

img_size = 28
channels = 1
num_classes = 10

y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true = tf.argmax(y_, axis=1)
x_ = tf.placeholder(tf.float32, shape=[None, img_size**2], name='x')

def net_model():
    #input layer
    input_layer = tf.reshape(x_, [-1, img_size, img_size, channels])
    #layer 1+
    conv1 = tf.layers.conv2d(input_layer, 32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    #layer 2
    conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    # fully connected layer
    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

    #final layer
    out = tf.layers.dense(inputs=dense, units=10)
    return tf.nn.softmax(out)

def net_train():
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=labels, labels=y_true)
    #average cross entropy
    cost = tf.reduce_mean(cross_entropy)
    #reduce cost of network
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    #calculate accuracy
    output_pred = tf.argmax(net_model, axis=1)
    correct_pred = tf.equal(output_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    train_batch_size = 64
    network = net_model()
    sess.run(tf.global_variables_initializer())
    x, y = mnist.train.next_batch(10)
    print("test")
    print(sess.run(network, feed_dict={x_:x, y_:y}))
