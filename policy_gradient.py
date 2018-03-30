import tensorflow as tf
import gym
import numpy as np
from replay_buffer import Memory
from matplotlib import pyplot as plt
import cProfile
import re
from tensorflow.python.client import timeline
import time as Time
from tensorflow.python import debug as tf_debug
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()



class Value:
    def __init__(self, env,scope):
        self.scope = scope
        self.env = env
        self.action_space = self.env.action_space.shape[0]
        self.obs_space = self.env.observation_space.shape[0]
        self.act_as_target = False
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)
            self.obs = tf.placeholder(tf.float32, shape=[None,self.obs_space], name="observation")
            self.action = tf.placeholder(tf.float32, shape=[None,self.action_space], name="action")
            self.target = tf.placeholder(tf.float32, shape=[None,1], name="target_value")
            self.value = self.value_model(self.obs, self.action)
            self.loss = -tf.reduce_mean(tf.squared_difference(self.value_model(self.obs,self.action),self.target))
            self.optimize_value = self.optimizer.minimize(self.loss)
            self.grad = tf.gradients(self.value, self.action)
    def create_target(self,TAU):
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        net = self.get_trainable_parameters()
        self.target_update_op = ema.apply(net)
        self.target_net_param = [ema.average(x) for x in net]
        # this_parm = self.get_trainable_parameters()
        # this_parm = target_net

    def value_model(self, obs, action):
        hidden = 64
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
            # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
            dense1 = tf.layers.dense(inputs=obs, units=hidden, activation=tf.nn.relu,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            dense1 = tf.contrib.layers.layer_norm(dense1, center=True, scale = True)
            dense1 = tf.concat([dense1, action], axis=1)
            dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            value = tf.layers.dense(inputs=dense2, units=1, activation=None,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            value = tf.contrib.layers.layer_norm(value, center=True, scale = True)
            return value
    def get_trainable_parameters(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            value_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            return value_parameters

    def get_all_parameters(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            value_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            return value_parameters

    def set_trainable_parameters(self,parameters,tau):
        this_policy_parameters = self.get_trainable_parameters()
        self.soft_assign = [tf.assign(v_t, v*tau+v_t*(tau-1)) for v_t, v in zip(this_policy_parameters,parameters)]
        sess = tf.get_default_session()
        sess.run(self.soft_assign)

    def get_q_gradient(self,action,obs):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            obs = obs.reshape(-1, self.obs_space)
            action = action.reshape(-1, self.action_space)
            sess = tf.get_default_session()

            return sess.run([self.grad],feed_dict={self.action:action,self.obs:obs})

    def predict(self, obs, action):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE,custom_getter=self.custom_getter):
            obs = obs.reshape(-1, self.obs_space)
            action = action.reshape(-1,self.action_space)
            sess = tf.get_default_session()
            value = sess.run([self.value], feed_dict={self.obs: obs, self.action: action})
            # tf.summary.scalar('value', value)
            return value

    def custom_getter(self,getter,name,*args, **kwargs):
        if self.act_as_target:
            return self.target_net_param
        else:
            return getter(name,*args, **kwargs)

    def update_value(self, obs, action, target):
        with tf.variable_scope(self.scope):
            sess = tf.get_default_session()
            feed_dict = {self.obs: obs, self.target: target, self.action:action}
            loss, _ = sess.run([self.loss,self.optimize_value], feed_dict)
            # print(loss)
    def update_target(self):
        sess = tf.get_default_session()
        sess.run(self.target_update_op)

class Policy:
    def __init__(self, env,scope):
        self.scope = scope
        self.act_as_target = False
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            self.env = env
            self.action_space = self.env.action_space.shape[0]
            self.obs_space = self.env.observation_space.shape[0]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
            self.obs = tf.placeholder(tf.float32, shape=[None, self.obs_space], name="observation")
            # self.target = tf.placeholder(tf.float32, shape=[None, self.action_space], name="target_action")
            self.action = self.policy_model(self.obs)
            self.Q_gradient = tf.placeholder(tf.float32, shape=[None, self.action_space ], name="Q_gradient")
            policy_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            policy_grad = tf.gradients(self.action, policy_parameters, self.Q_gradient)
            self.policy_step = self.optimizer.apply_gradients(zip(policy_grad, policy_parameters))
    def noise(self, mu=None, sigma=None):
        if mu is None:
            mu = np.zeros([1,self.action_space])
        if sigma is None:
            sigma = 0.00001
        return np.random.normal(mu, sigma * np.ones([1,self.action_space]))

    def create_target(self,TAU):
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        net = self.get_trainable_parameters()
        self.target_update_op = ema.apply(net)
        self.target_net_param = [ema.average(x) for x in net]

    def update_target(self):
        sess = tf.get_default_session()
        sess.run(self.target_update_op)

    def policy_model(self, obs):
        hidden = 64
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
            # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
            dense1 = tf.layers.dense(inputs=obs, units=hidden, activation=tf.nn.relu,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # dense1 = tf.contrib.layers.layer_norm(dense1, center=True, scale = True)
            dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu)
            # dense2 = tf.contrib.layers.layer_norm(dense2, center=True, scale=True)
            action = tf.layers.dense(inputs=dense2, units=self.action_space,activation=tf.nn.tanh,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            action = tf.clip_by_value(action,self.env.action_space.low,self.env.action_space.high)
            return action

    def custom_getter(self,getter,name,*args, **kwargs):
        if self.act_as_target:
            return self.target_net_param
        else:
            return getter(name,*args, **kwargs)

    def predict(self, obs):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE,custom_getter=self.custom_getter):
            obs = obs.reshape(-1,self.obs_space)
            sess = tf.get_default_session()
            return sess.run([self.action], feed_dict={self.obs: obs}) + self.noise()


    def get_trainable_parameters(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            policy_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            return policy_parameters

    def set_trainable_parameters(self,parameters,tau):
        this_policy_parameters = self.get_trainable_parameters()
        self.soft_assign = [tf.assign(p_t, p*tau+p_t*(tau-1)) for p_t, p in zip(this_policy_parameters,parameters)]
        sess = tf.get_default_session()
        sess.run(self.soft_assign)
        # sess = tf.get_default_session()
        # sess.run(self.target_update)

    def optimize_policy(self,q_grad,obs):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            obs = obs.reshape(-1, self.obs_space)


            # print(policy_parameters)
            # loss = tf.reduce_mean(policy_grad)
            temp = set(tf.all_variables())

            sess = tf.get_default_session()

            # sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
            sess.run([self.policy_step],feed_dict={self.Q_gradient:q_grad,self.obs:obs})
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_01.json', 'w') as f:
            #     f.write(chrome_trace)
            # tf.summary.scalar('loss', tf.reduce_mean(loss))
            # merged = tf.summary.merge_all()

def train():
    gamma = 0.99
    episodes = 100
    batch_size = 128
    max_time_steps = 200
    episode_reward = 0
    reward_history = []
    env = gym.make('Pendulum-v0')
    obs_old = env.reset()
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    agent_policy = Policy(env,"policy")
    agent_critic = Value(env,"value")
    # agent_policy_t = Policy(env,"policy_t")
    # agent_critic_t = Value(env,"value_t")

    #initial rollouts to gather date
    for i in range(10000):
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        episode_reward += rew

        memory.append(obs_old, action, rew, obs, done)
        obs_old = obs
        if done:
            # reward_history.append(episode_reward)
            episode_reward = 0
            env.reset()
    episode_reward = 0
    time_t = 200
    tf.summary.scalar("episode_time_steps", time_t)
    tf.summary.scalar("episode_reward", episode_reward)
    merged = tf.summary.merge_all()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    mark = np.zeros([8,1])
    time_split = np.zeros_like(mark)
    with tf.Session() as sess:
        sess1 = tf_debug.TensorBoardDebugWrapperSession(sess, "Vader:6007")
        # tf_debug.LocalCLIDebugWrapperSession(sess)
        from datetime import datetime
        now = datetime.now()
        train_writer = tf.summary.FileWriter('./train/'+now.strftime("%Y%m%d-%H%M%S")+'/', sess.graph)
        # train_writer = tf.summary.FileWriter('.' + '/train', sess.graph)

        agent_critic.create_target(0.1)
        agent_policy.create_target(0.1)
        # agent_policy_t.create_target_capacity(agent_policy.get_trainable_parameters(),0.6)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        sess1.run(agent_policy.get_trainable_parameters())
        # agent_critic.set_trainable_parameters(agent_critic.get_trainable_parameters(), 0)
        # agent_policy.set_trainable_parameters(agent_policy.get_trainable_parameters(), 0)
        for i in range(episodes):
            print('running episode:',i)
            t = 0
            done = 0

            while t < max_time_steps:
                # print(t)
                start = Time.time()
                action = agent_policy.predict(obs_old)# + agent_policy.noise(0,1/episode_reward)
                mark[0] = Time.time() - start
                # print('mark1:'+str(mark1))
                action = action.reshape(-1)
                # if action>0.5:
                #     action = 1
                # else:
                #     action = -0
                # action = env.action_space.sample()
                obs, rew, done, info = env.step(action)
                # env.render()
                episode_reward += rew
                memory.append(obs_old, action, rew, obs, done)

                # print('mark2:' + str(mark2))
                if done or t == max_time_steps-1:
                    time_t = t
                    reward_history.append(episode_reward)
                    episode_reward = 0
                    env.reset()
                obs_old = obs
                t += 1

            for steps in range(50):
                batch = memory.sample(batch_size)
                obs_batch = batch['obs0']
                obs_batch -= np.mean(obs_batch,0)
                obs_batch = obs_batch/np.var(obs_batch,0)
                agent_policy.act_as_target = True
                action_batch_predict = agent_policy.predict(obs_batch)[0]
                agent_policy.act_as_target = False
                agent_critic.act_as_target = True
                value_batch = agent_critic.predict(obs_batch,action_batch_predict)[0]
                # print(value_batch[0])
                agent_critic.act_as_target = False
                y = np.array(batch['rewards']) + gamma* np.array(value_batch)#.reshape(-1,batch_size)
                agent_critic.update_value(obs= obs_batch,action = action_batch_predict,target=y)
                q_grad = np.array(agent_critic.get_q_gradient(action_batch_predict, obs_batch)).reshape(-1,env.action_space.shape[0])
                agent_policy.optimize_policy(q_grad,obs_batch)
                parm = agent_critic.get_all_parameters()
                value = np.array(sess.run(agent_critic.get_all_parameters()))
                agent_critic.update_target()
                agent_policy.update_target()
                value = np.array(sess.run(agent_critic.get_all_parameters()))

            # print(time_split)
            print(reward_history[-1])
            summary = sess.run(merged)
            train_writer.add_summary(summary, i)
            time = 200
            episode_reward = 0

    # print(reward_history)
    # plt.plot(reward_history)


train()