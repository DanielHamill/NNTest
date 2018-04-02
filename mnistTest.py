import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sys import platform

if platform == "win32":
    import win_unicode_console as wc
    wc.enable()

mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

n_epochs = 10
hl_nodes_1 = 500
hl_nodes_2 = 500
hl_nodes_3 = 500

n_classes = 10
batch_size = 100

# placeholder variables for x and y (in and out)
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32)

def net_model():
    weight1 = tf.Variable(tf.random_uniform([784, hl_nodes_1], -1, 1))
    bias1 = tf.Variable(tf.zeros([hl_nodes_1]))

    weight2 = tf.Variable(tf.random_uniform([hl_nodes_1, hl_nodes_2], -1, 1))
    bias2 = tf.Variable(tf.zeros([hl_nodes_2]))

    weight3 = tf.Variable(tf.random_uniform([hl_nodes_2, hl_nodes_3], -1, 1))
    bias3 = tf.Variable(tf.zeros([hl_nodes_3]))

    weightOut = tf.Variable(tf.random_uniform([hl_nodes_3, n_classes], -1, 1))
    biasOut = tf.Variable(tf.zeros([n_classes]))

    layer1 = tf.matmul(x_, weight1) + bias1
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.matmul(layer1, weight2) + bias2
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.matmul(layer2, weight3) + bias3
    layer3 = tf.nn.relu(layer3)

    output = tf.matmul(layer3, weightOut) + biasOut
    return output

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
            _, c = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})

            total_cost += c

        print('Epoch', epoch, 'completed out of', n_epochs,'loss:', total_cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:',accuracy.eval({x_:mnist.test.images, y_:mnist.test.labels}, session=sess))

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

with tf.Session() as sess:
    n_epochs = 10
    prediction = net_model()
    train_net(sess)

    while input('press enter')!='end':
        # feed_net_mnist(sess)
        fil = 'images/digit' + input('input digit') + '.png'
        feed_net_file(sess,fil)
