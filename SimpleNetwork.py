import tensorflow as tf

INPUTS = 2
HIDDEN_NODES = 5
OUTPUTS = 2

X_ = tf.placeholder(tf.float32, [, INPUTS], 'Input')
output = tf.placeholder(tf.float32), [1,OUTPUTS])

w1 = tf.variable(tf.random_uniform([IPUTS,HIDDEN_NODES], -1, 1), name="weights 1")
