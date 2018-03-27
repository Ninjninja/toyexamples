import tensorflow as tf
import math as math
import numpy as np
from tensorflow.python.framework import ops
class model:
    def __init__(self):
        self.create_model()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    def create_model(self):
        with tf.variable_scope("model"):
            self.input = tf.placeholder(dtype=tf.float32,shape=[None,10])
            self.out = tf.layers.dense(inputs=self.input,units=1,activation=None,kernel_initializer=tf.random_normal_initializer,bias_initializer=tf.zeros_initializer)
    def predict(self,input):
        with tf.variable_scope("model",reuse=tf.AUTO_REUSE):
            sess = tf.get_default_session()
            init = tf.initialize_all_variables()
            # sess.run(init)
            return sess.run([self.out],feed_dict={self.input:input})[0]

    def optimize(self,input,output):
        sess = tf.get_default_session()
        with tf.variable_scope("model",reuse=tf.AUTO_REUSE):
            variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,scope="model")
            grad = tf.gradients(self.out,variables)

            grad_out_in = sess.run([grad],feed_dict={self.input:input})
            print(grad_out_in,variables)
            train_ops =self.optimizer.apply_gradients(zip(grad,variables))
            sess.run([train_ops],feed_dict={self.input:input})

with tf.Session() as sess:
    # init = tf.ini
    # sess.run(init)
    new = model()
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    init = tf.variables_initializer(all_variables_list)
    print('variables')
    print(all_variables_list)
    sess.run(init)

    out = (new.predict(input=np.ones([5,10])))
    new.optimize(input=np.ones([5,10]),output=np.array([1]).reshape(-1,1))