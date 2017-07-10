# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import ndimage as img
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("./tmp/MNIST_data/", one_hot=True) #for windows users
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(X, W) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))



# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
fig1 = plt.figure()
plt.imshow(mnist.test.images[r:r + 1].
           reshape(28, 28), cmap='Greys', interpolation='nearest')


print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))


'''
img1 = mnist.test.images[r:r + 1]
x = np.array(img1)
print x.shape   # (1, 784)
'''


x = img.imread("data/9_28x.png")
x = x.astype(float)
x1 = np.array(x)
x1 = x1/255.0;
x1 = 1.0 - x1;
x1_flat = x1.reshape(1,784 )

print x1_flat.shape   # (1, 784)

print("Label: ", 9)
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X:x1_flat}))
fig2 = plt.figure()
plt.imshow(x1_flat.
           reshape(28, 28), cmap='Greys', interpolation='nearest')



x = img.imread("data/4_28x.png")
x = x.astype(float)
x1 = np.array(x)
x1 = x1/255.0
x1 = 1.0 - x1
x1_flat = x1.reshape(1,784 )

print x1_flat.shape   # (1, 784)

print("Label: ", 4)
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X:x1_flat}))
fig3 = plt.figure()
plt.imshow(x1_flat.
           reshape(28, 28), cmap='Greys', interpolation='nearest')

plt.show()

