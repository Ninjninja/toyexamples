import tensorflow as tf
import math as math
import numpy as np
from tensorflow.python.framework import ops
class model:
    def __init__(self,scope):
        self.scope = scope
        self.create_model()
        ema = tf.train.ExponentialMovingAverage(decay=1 - 0.001)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    def create_model(self):
        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(dtype=tf.float32,shape=[None,10])
            self.out = tf.layers.dense(inputs=self.input,units=1,activation=None,kernel_initializer=tf.random_normal_initializer(0,1e-3),bias_initializer=tf.zeros_initializer)
    def predict(self,input):
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            sess = tf.get_default_session()
            init = tf.initialize_all_variables()
            # sess.run(init)
            return sess.run([self.out],feed_dict={self.input:input})[0]

    def create_target(self,TAU):
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        self.maintain_avg_op = ema.apply(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,scope=self.scope))
        # # self.target_update = ema.apply(net)
        # target_net = [ema.average(x) for x in net]
        # this_parm = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,scope=self.scope)
        # self.init_assign = [tf.assign(x,y) for x,y in zip(this_parm,target_net)]

    def optimize(self,input,output):
        sess = tf.get_default_session()
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            variables = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,scope=self.scope)
            grad = tf.gradients(self.out,variables)

            grad_out_in = sess.run([grad],feed_dict={self.input:input})
            print(grad_out_in,variables)
            train_ops =self.optimizer.apply_gradients(zip(grad,variables))
            sess.run([train_ops],feed_dict={self.input:input})

with tf.Session() as sess:
    # init = tf.ini
    # sess.run(init)
    new = model("model")
    # new_t = model("t_model")
    new.create_target(0.99)
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    init = tf.variables_initializer(all_variables_list)
    print('variables')
    print(all_variables_list)
    sess.run(init)
    out = (new.predict(input=np.ones([5,10])))
    values = np.array(sess.run(all_variables_list))
    new.optimize(input=np.ones([5,10]),output=np.array([1]).reshape(-1,1))
    values = np.array(sess.run(all_variables_list))
    sess.run(new.maintain_avg_op)
    values = np.array(sess.run(all_variables_list))
    print('abcd')