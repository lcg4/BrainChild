# Solves XOR
# Gets cost REALLY low to like 1e-12

import tensorflow as tf
import time

x = tf.placeholder(tf.float32, shape = [4, 2], name = "x-input")
y = tf.placeholder(tf.float32, shape = [4, 1], name = "y-input")

th1 = tf.Variable(tf.random_uniform([2, 2], -1, 1, name = "Theta1"))
th2 = tf.Variable(tf.random_uniform([2, 1], -1, 1, name = "Theta2"))

bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

A2 = tf.sigmoid(tf.matmul(x, th1) + bias1)
hypothesis = tf.sigmoid(tf.matmul(A2, th2) + bias2)

cost = tf.square(y - hypothesis)

step = tf.train.AdamOptimizer(0.01).minimize(cost)

XORX = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
XORY = [ [0], [1], [1], [0] ]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

feed_dict = { x : XORX, y : XORY }

# TRAIN BABY TRAIN
t0 = time.time()
for i in range(100000):
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
        print('cost ',
            sess.run(cost, feed_dict = feed_dict))
t1 = time.time()

print(t1 - t0)
