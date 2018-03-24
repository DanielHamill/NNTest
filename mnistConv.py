#https://www.tensorflow.org/tutorials/layers

import tensorflow as tf
import numpy as np

def net_model():
    #input layer
    input_layer = tf.reshape(tensor, [-1, 28, 28, 1])
    #layer 1
    conv1 = tf.layers.conv2d(input_layer, 32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    #layer 2
    conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    # fully connected layer
    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    #dropout layer (for overfitting?)
    dropout = tf.layers.dropout(dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #final layer
    logits = tf.layers.dense(input=dropout, units=10)

    predictions = {
        'classes':tf.arg_max(input=logits, axis=1)
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #loss function: using cross entropy
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
