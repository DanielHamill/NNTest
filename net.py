import tensorflow as tf
import numpy as np
from sys import platform

if platform == "win32":
	import win_unicode_console as wc
	wc.enable()

x = [[0, 0],[0, 1],[1, 0],[1, 1]]
y = [[0], [1], [1], [0]]

N_STEPS = 30000
N_EPOCH = 500
N_TRAINING = len(x)
LEARNING_RATE = 0.05

N_INPUT_NODES = 2
N_HIDDEN_NODES = 5
N_OUTPUT_NODES = 1

y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="output")

x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="input")
weight1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="weight1")
weight2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_OUTPUT_NODES], -1, 1), name="weight2")

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

	# training
	sess.run(train_step, feed_dict={x_: x, y_: y})

	if j % N_EPOCH == 0:
			print('Run Number ', j)
			print('Ouput ', sess.run(output, feed_dict={x_: x, y_: y}))
			print('Cost ', sess.run(cost, feed_dict={x_: x, y_: y}))

print('Press Enter')

while raw_input('') != 'no':

	userx1 = raw_input("input 1: ")
	userx2 = raw_input("input 2: ")

	user_input = [[userx1, userx2],[0, 0],[0, 0],[0, 0]]

	fl = float(sess.run(output, feed_dict={x_: user_input, y_: y})[0][0])
	rounded = round(fl)
	integerized = int(rounded)
	out = str(integerized)

	print('output: ' + out)

	print('----------')
	print('continue?')
	print('----------')
