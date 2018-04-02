import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from sys import platform

#some things are different depending on windows vs mac
if platform == "win32":
    import win_unicode_console as wc
    wc.enable()
    clear = lambda: os.system('cls')
else:
    clear = lambda: os.system('clear')
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

n_classes = 10
batch_size = 100

# placeholder variables for x and y (in and out)
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32)
p_ = tf.placeholder_with_default(0.0, shape=())

#parameters for dropout
drop_rate = 0.5

def conv_model():
    #input layer
    input_layer = tf.reshape(x_, [-1, 28, 28, 1])
    #layer 1+
    conv1 = tf.layers.conv2d(input_layer, 32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    #layer 2
    conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    # fully connected layer
    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

    #dropout layer for overfitting correction
    dropout = tf.layers.dropout(inputs=dense, rate=p_, training=True)

    #final layer
    out = tf.layers.dense(inputs=dropout, units=10)
    return tf.nn.softmax(out)

def train_net(sess):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y_))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print("start training")
    # with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        total_cost = 0
        for current_batch in range(int(mnist.train.num_examples/batch_size)):
            x, y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x_: x, y_: y, p_: drop_rate})

            total_cost += c

        print('Epoch', epoch, 'completed out of', n_epochs,'loss:', total_cost)

    # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    # print('Accuracy:',accuracy.eval({x_:mnist.test.images, y_:mnist.test.labels}, session=sess))

def feed_net_mnist(sess):
    x = int(input('input index:'))

    digit = mnist.test.images[x]
    img = np.array(digit)
    pixels = img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    # with tf.Session() as sess:
    result = tf.argmax(prediction, 1)
    print(sess.run(result, feed_dict={x_: [digit], y_: [mnist.test.labels[x]]}))

def feed_net_file(sess, file_path):
    raw = img.imread(file_path)
    rgba = np.array(raw)
    greyscale = []
    for i in range(len(rgba)):
        for j in range(len(rgba[0])):
            greyscale.append(rgba[i][j][0])
    greyscale = np.array(greyscale)

    # print(greyscale)

    pixels = greyscale.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    # with tf.Session() as sess:
    result = tf.argmax(prediction, 1)
    print(sess.run(result, feed_dict={x_: [greyscale], y_: [mnist.test.labels[0]]}))
    return

def get_input(str):
    if platform == "win32":
        return input(str)
    else:
        return raw_input(str)

with tf.Session() as sess:
    #clear screen
    clear()

    #number of epochs
    n_epochs = 10
    prediction = conv_model()
    #saves current session
    saver = tf.train.Saver()

    #saves session if user retrains net
    if(get_input('retrain net?') == 'y'):
        train_net(sess)
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
    else:
    #restores session if user does not retrain net
        saver.restore(sess, "/tmp/model.ckpt")
        print('restoring network saved in /tmp/model.ckpt')

    training = False

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:',accuracy.eval({x_:mnist.test.images, y_:mnist.test.labels}, session=sess))

    while get_input('press enter')!='end':
        clear()
        # feed_net_mnist(sess)
        # fil = 'images/digit' + input('input digit') + '.png'
        fil = 'images/' + get_input('input digit') + '.png'
        feed_net_file(sess,fil)
