import tensorflow as tf
from tensorflow.contrib import rnn
import tensorboard
#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#define constants
#unrolled through 28 time steps
time_steps=28
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=28
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=10
#size of batch
batch_size=128
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))
x = tf.placeholder(tf.float32,[None,time_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

input = tf.unstack(x,time_steps,axis=1)
lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer,input,dtype=tf.float32)
print('out')
prediction = tf.layers.dense(inputs=outputs[-1],units=10)
# prediction=tf.matmul(outputs[-1],out_weights)+out_bias
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#correct prediction
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
tf.summary.scalar('loss',loss)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
with tf.Session() as sess:

    train_writer = tf.summary.FileWriter('./train', sess.graph)
    sess.run(init)
    for i in range(1000):
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)

        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        if i %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los,summary_out=sess.run([loss,merged],feed_dict={x:batch_x,y:batch_y})
            train_writer.add_summary(summary_out, i)
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")