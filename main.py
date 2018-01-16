# Solves XOR
# Gets loss REALLY low to like 1e-12

import tensorflow as tf
import time

XORX = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
XORY = [ [0], [1], [1], [0] ]

x = tf.placeholder(tf.float32, shape = [4, 2], name = "x-input")
y = tf.placeholder(tf.float32, shape = [4, 1], name = "y-input")

th1 = tf.Variable(tf.random_uniform([2, 2], -1, 1, name = "Theta1"))
th2 = tf.Variable(tf.random_uniform([2, 1], -1, 1, name = "Theta2"))

bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

A2 = tf.sigmoid(tf.matmul(x, th1) + bias1)

hypothesis = tf.sigmoid(tf.matmul(A2, th2) + bias2)

cross_entropy = tf.square(y - hypothesis)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(0.01)

step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# TRAIN BABY TRAIN
t0 = time.time()
for i in range(100000):
    feed_dict = { x : XORX, y : XORY }
    sess.run(step, feed_dict = feed_dict)
    if(i % 1000 == 0):
        print('Hypothesis ',
            sess.run(hypothesis, feed_dict = feed_dict))
        print('Theta1 ',
            sess.run(th1))
        print('Bias1 ',
            sess.run(bias1))
        print('Theta2 ',
            sess.run(th2))
        print('Bias2 ',
            sess.run(bias2))
        print('loss',
            sess.run(loss, feed_dict = feed_dict))
t1 = time.time()

print(t1 - t0)
