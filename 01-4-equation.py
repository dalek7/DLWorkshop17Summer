#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

#A = tf.placeholder(tf.float32, [2, 2])
#b = tf.placeholder(tf.float32, [2, 1])

# 중학생 연립방정식 풀기

A = [[1.0, 2.0],
     [1.0, -3.0]]

b = [[6.0],
     [1.0]];

# 정답. 이게 나와야함
sol = [[4.0],
     [1.0]]

bb = tf.matmul(A, sol)
print bb


x = tf.Variable(tf.random_normal([2, 1]), name='weight1')

cost = tf.reduce_mean(tf.square(tf.matmul(A, x) - b))

learning_rate = 0.1
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Launch the default graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    bb1 =  sess.run(bb)
    #print np.array(bb1).shape
    #print np.array(sess.run(x)).shape
    for step in range(1000):
        sess.run(train)


    x1= sess.run(x)
    print('estimated =')
    print(x1)
    print('sol =')
    print(sol)

