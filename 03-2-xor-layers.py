# XOR
# based on https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-2-xor-nn.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1


x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

y_data = [[0],
          [1],
          [1],
          [0]]


x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

#x1 = x_data[[0], :]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

    W1val = sess.run(W1)
    W2val = sess.run(W2)
    b1val = sess.run(b1)
    b2val = sess.run(b2)

    print ('W1 = ', sess.run(W1))
    print ('W2 = ', sess.run(W2))
    print ('b1 = ', sess.run(b1))
    print ('b2 = ', sess.run(b2))

    for i in range(4):
        x1 = x_data[[i], :]
        #print x1
        l1 = tf.sigmoid(tf.matmul(x1, W1) + b1)
        l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)
        l2cast = tf.cast(l2 > 0.5, dtype=tf.float32)
        print i, sess.run(l2), sess.run(l2cast), y_data[[i], :]


'''
W1 = np.array([[ 6.26347065,  6.12451124],   [-6.38764334, -5.81880665]])
W2 = np.array([[ 10.10004139], [ -9.59866238]])
b1 = np.array([-3.40124607,  2.879565  ])
b2 = np.array([4.46212006])
'''
	   
'''
('W1 = ', array([[ 6.26347065,  6.12451124],
       [-6.38764334, -5.81880665]], dtype=float32))
('W2 = ', array([[ 10.10004139],
       [ -9.59866238]], dtype=float32)

0 [[ 0.01338216]] [[ 0.]] [[ 0.]]
1 [[ 0.98166394]] [[ 1.]] [[ 1.]]
2 [[ 0.98809403]] [[ 1.]] [[ 1.]]
3 [[ 0.01135799]] [[ 0.]] [[ 0.]]

'''