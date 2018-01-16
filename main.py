# Solves XOR

import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [4, 2], name = "x-input")
y = tf.placeholder(tf.float32, shape = [4, 1], name = "y-input")

th1 = tf.Variable(tf.random_uniform([2, 2], -1, 1, name = "Theta1"))
th2 = tf.Variable(tf.random_uniform([2, 1], -1, 1, name = "Theta2"))

bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

A2 = tf.sigmoid(tf.matmul(x, th1) + bias1)
hypothesis = tf.sigmoid(tf.matmul(A2, th2) + bias2)

cost = tf.reduce_mean(((y * tf.log(hypothesis)) + ((1 - y) * tf.log(1.0 - hypothesis))) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XORX = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
XORY = [ [0], [1], [1], [0] ]

init = tf.global_variables_initializer()
sess = tf.Session()

# Set to True to save graph output
if False:
    writer = tf.summary.FileWriter("xor_logs", sess.graph)

sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict = { x : XORX, y : XORY })
    if(i % 1000 == 0):
        print('Hypothesis ',
            sess.run(hypothesis, feed_dict = { x : XORX, y : XORY }))
        print('Theta1 ',
            sess.run(th1))
        print('Bias1 ',
            sess.run(bias1))
        print('Theta2 ',
            sess.run(th2))
        print('Bias2 ',
            sess.run(bias2))
        print('cost ',
            sess.run(cost, feed_dict = { x : XORX, y : XORY}))
