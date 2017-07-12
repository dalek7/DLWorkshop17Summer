# based on https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
#sess = tf.Session()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# First Convolutional Layer
## first layer will consist of convolution, followed by max pooling
## The convolution will compute 32 features for each 5x5 patch.
## Its weight tensor will have a shape of [5, 5, 1, 32].
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

## reshape x to a 4d tensor
x_image = tf.reshape(x, [-1,28,28,1])

## We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## The max_pool_2x2 method will reduce the image size to 14x14.
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
## The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) ## the image size will be reduced to 7x7

# Densely Connected Layer
n_neurons = 512;
## Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
## We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, n_neurons])
b_fc1 = bias_variable([n_neurons])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
#W_fc2 = weight_variable([1024, 10])
W_fc2 = weight_variable([n_neurons, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(1000): # 20000
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g"% sess.run(
        accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()

# Ran out of memory trying to allocate 957.03MiB.  See logs for memory state !!
# TODO : upgrade GPU !

