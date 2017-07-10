#-*- coding: utf-8 -*-
# Lab 3 Logistic Regression Classifier
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# 좌표
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

# Label
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
# linear의 경우
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

steps = []
costs = []
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        steps.append(step)
        costs.append(cost_val)

        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    print sess.run(W)

    for i in range(len(x_data)):
        v1 = sess.run(hypothesis, feed_dict={X: [x_data[i]]})
        v2 = round(v1)



        print x_data[i], "--->",  v1, "-->", v2;
        #print(sess.run(hypothesis, feed_dict={X: [[1, 2]]}))
        #print(sess.run(hypothesis, feed_dict={X: [[1, 2]]}))

    x_data2 = np.array(x_data)
    x1 = x_data2[:, 0]
    x2 = x_data2[:, 1]
    print len(x1);
    print ('x1=', x1)
    print ('x2=', x2)

    plt.figure(0)
    for i in range(len(x1)):
        if y_data[i] == [0]:
            plt.plot(x1[i], x2[i], "bo")
        else:
            plt.plot(x1[i], x2[i], "r^")


    plt.show()
