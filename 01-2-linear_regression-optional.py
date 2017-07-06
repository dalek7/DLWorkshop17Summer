import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
import numpy as np
import matplotlib.pyplot as plt

# X and Y data
x_train = [1, 2, 3]
#y_train = [2, 4, 6]
y_train = [2+0.1, 4-0.3, 6+0.15]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 2 and b should be 0
# But let TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

vcost =[]
# Fit the line
#plt.figure(1)

plt.figure(1)
plt.plot(x_train, y_train, 'o')
plt.grid()
plt.axis((np.min(x_train) - 1, np.max(x_train) + 1, np.min(y_train) - 1, np.max(y_train) + 1))
x1 =np.linspace(np.min(x_train)-1, np.max(x_train)+1, num=5)


vm1 = []
vb1 = []


for step in range(2001):
    sess.run(train)
    vcost.append(sess.run(cost))
    vm1.append(sess.run(W)[0])
    vb1.append(sess.run(b)[0])

    if step % 20 == 0:
        m1 = sess.run(W)[0]
        b1 = sess.run(b)[0]
        print(step, sess.run(cost), m1, b1)
        plt.plot(x1, m1 * x1 + b1, color=(0.8,0.8,0.8,0.2))
        plt.pause(0.05)

plt.plot(x1, m1 * x1 + b1, 'm')
plt.plot(x_train, y_train, 'bo')

plt.figure(2)
plt.plot(vcost)


f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot(vm1)
axarr[0].set_title('W')
axarr[1].plot(vb1)
axarr[1].set_title('b')
plt.show()