import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    w           = tf.Variable(tf.random_normal([3, 3]), name='w')
    sess.run(tf.global_variables_initializer())
    print ('w = ', sess.run(w))
    sess.close()