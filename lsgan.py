import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 1000
learning_rate = 1e-4
epoch = 10000

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])


def generator():
    z = tf.random_uniform(shape=[batch_size, 100], minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
    fc1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=7*7*128, activation_fn=None)
    fc1 = tf.reshape(fc1, shape=[batch_size, 7, 7, 128])
    conv1 = tf.contrib.layers.conv2d_transpose(fc1, num_outputs=128, kernel_size=5, stride=2, padding="SAME", activation_fn=tf.nn.relu)
    conv2 = tf.contrib.layers.conv2d_transpose(conv1, num_outputs=128, kernel_size=5, stride=1, padding="SAME", activation_fn=tf.nn.relu)
    conv3 = tf.contrib.layers.conv2d_transpose(conv2, num_outputs=1, kernel_size=5, stride=2, padding="SAME", activation_fn=tf.nn.tanh)

    return conv3


def discriminator(tensor):

    conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=5, stride=2, padding="SAME",activation_fn=tf.nn.relu)
    conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=5, stride=2, padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm)
    fc1 = tf.reshape(conv2, shape=[batch_size, 7*7*64])
    fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=7*7*64, activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm)
    fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, activation_fn=tf.nn.relu)

    return fc2


g_out = generator()
d_out_fake = discriminator(g_out)
d_out_real = discriminator(x_image)

# loss & optimizer

disc_loss = tf.reduce_sum(tf.square(d_out_real-1) + tf.square(d_out_fake))
gen_loss = tf.reduce_sum(tf.square(d_out_fake-1))
total_loss = disc_loss + gen_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        _, a, gen_out, d_fake = sess.run([optimizer, total_loss, g_out, disc_loss], feed_dict={x: batch[0]})
        print("{} th fake: {}".format(i, np.sum(d_fake)/batch_size))

        if i % 1000 == 0:

            gen_o = sess.run(g_out)
            plt.imshow(gen_o[0][:, :, 0], cmap="gray")
            plt.show()
