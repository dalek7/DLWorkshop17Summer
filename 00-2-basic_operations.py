import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = a+b #tf.add(a, b)
mul = a*b #tf.mul(a, b)
# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
