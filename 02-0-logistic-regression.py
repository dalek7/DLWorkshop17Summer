#-*- coding: utf-8 -*-
# Lab 3 Logistic Regression
# Seung-Chan Kim
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777)  # for reproducibility


x1 = [0, 1, 2, 3, 4, 5];
x2 = [0, 0, 0, 1, 1, 1];

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(X * W + b)
# slope ?

# cost/loss function

#cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


steps = []
costs = []
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x1, Y: x2})
        steps.append(step)
        costs.append(cost_val)

        if step % 200 == 0:
            print(step, cost_val)

    for i in range(len(x1)):
        v1 = sess.run(hypothesis, feed_dict={X: [x1[i]]})
        print x1[i], "--->", v1

    plt.figure(0)
    plt.plot(x1, x2, "o")

    vy = []
    vx = []
    for i in range(100):
        xtmp =6.0*i/100
        v1 = sess.run(hypothesis, feed_dict={X: [xtmp]})
        vy.append(v1)
        vx.append(xtmp)
    plt.plot(vx,vy)


    plt.figure(1)
    plt.plot(steps, costs)
    plt.title('cost', fontsize=10)
    plt.show()

