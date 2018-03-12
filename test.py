import tensorflow as tf
#import win_unicode_console as wc
#wc.enable()

x = [[0, 0],[0, 1],[1, 0],[1, 1]]
y = [[0], [1], [1], [0]]

N_STEPS = 30000
N_EPOCH = 500
N_TRAINING = len(x)
LEARNING_RATE = 0.05

N_INPUT_NODES = 2
N_HIDDEN_NODES = 5
N_OUTPUT_NODES = 1

x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="input")
y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="output")

#weight1 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_INPUT_NODES], -1, 1), name="weight1")
weight1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="weight1")
weight2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_INPUT_NODES], -1, 1), name="weight2")
#weight2 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="weight2")

bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")

layer1 = tf.sigmoid(tf.matmul(x_, weight1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, weight2) + bias2)

cost = tf.reduce_mean(tf.square(y - output))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for j in range(N_STEPS):

	sess.run(train_step, feed_dict={x_: x, y_: y})

	if j % N_EPOCH == 0:
			print('Run Number ', j)
			print('Training ', sess.run(output, feed_dict={x_: x, y_: y}))
			print('Cost ', sess.run(cost, feed_dict={x_: x, y_: y}))



userx1 = input("input")
userx2 = input("input")

user_input = [[userx1, userx2]]

print('ouput, ', sess.run(output, feed_dict={x_: user_input, y_: y}))
