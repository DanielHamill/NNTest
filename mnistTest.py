import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

def train_net():
    prediction = net_model()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("start")

        for epoch in range(n_epochs):
            total_cost = 0
            for current_batch in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x_: x, y_: y})

                total_cost += c

            print('Epoch', epoch, 'completed out of', n_epochs,'loss:', total_cost)


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x_:mnist.test.images, y_:mnist.test.labels}))



train_net()
