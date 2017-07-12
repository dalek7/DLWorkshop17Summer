# Seung-Chan Kim

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = input_data.read_data_sets("./tmp/MNIST_data/", one_hot=True) #for windows users

idx_to_test = 50
img     = mnist.train.images[idx_to_test]
label   = mnist.train.labels[idx_to_test]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#print img
print(label)  # 5

print(img.shape)  # (784,)
img_flat = img.reshape(28, 28)
print(img_flat.shape)  # (28, 28)

# img_show(img)

fig3 = plt.figure()
plt.imshow(img_flat.
           reshape(28, 28), cmap='Greys', interpolation='nearest')

label1 = label.astype(int)
str1 = " ".join(str(x) for x in label1)
ttl = '#%d      [%s]' % (idx_to_test, str1)
plt.title(ttl, fontsize=15)

plt.show()
