# coding: utf-8
# based on https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/functions.py

import numpy as np
import tensorflow as tf

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([-3.0, 1.1, -0.8, 4.0])
y = softmax(x)
y2 = tf.nn.softmax(x)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(x)
print(y)
print(sess.run(y2))

print('sum (y)=', np.sum(y))
print('sum (y2)=', np.sum(sess.run(y2)))