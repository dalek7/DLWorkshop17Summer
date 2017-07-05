import tensorflow as tf
import numpy as np

x = tf.placeholder("float", None)
relu_out = tf.nn.relu(x)

# Launch the default graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        relu_out1 = sess.run(relu_out, feed_dict={x:i-5})
        print (i-5, '-->', relu_out1)

    relu_out1 = sess.run(relu_out, feed_dict={x: np.array([[-1.0, 2.0, 3.0]])})
    print relu_out1
