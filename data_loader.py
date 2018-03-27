import tensorflow as tf
import tensorboard
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
# mnist = mnist.train.images.reshape(-1, 28, 28)
input_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
# conv1 = tf.layers.conv2d(inputs=input_image, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
# pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[4,4],strides=3)
# conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
# pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[3,3],strides=2)
# dense1 = tf.layers.dense(inputs=tf.layers.flatten(pool2),units=256)
# dense2 = tf.layers.dense(inputs=dense1,units=10)
conv1 = rnn.Conv2DLSTMCell()
conv1 = tf.layers.conv2d(inputs=input_image, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[4,4],strides=3)
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[3,3],strides=2)
dense1 = tf.layers.dense(inputs=tf.layers.flatten(pool2),units=256)
dense2 = tf.layers.dense(inputs=dense1,units=10)

prediction = tf.nn.softmax(dense2)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense2,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()
tf.summary.scalar('loss',loss_op)
merged = tf.summary.merge_all()


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/tmp' + '/train', sess.graph)
    sess.run(init)
    for i in range(100000):
        batch_x, batch_y = mnist.train.next_batch(1)
        batch_x = batch_x.reshape(-1,28,28,1)
        # print(batch_x.shape)
        # print(batch_y.shape)
        _,loss,merge = sess.run([train_op,loss_op,merged], feed_dict={input_image:batch_x,Y:batch_y})
        train_writer.add_summary(merge,i)
        if not i%100:
            print(loss)


