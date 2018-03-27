import tensorflow as tf
import gym
import numpy as np
class CartPoleAgent:
    def __init__(self,env,memory):
        self.env = env
        self.policy = CartPoleAgent.Policy(env)
        self.critic = CartPoleAgent.Value(env)
        self.memory = memory
        self.gamma = 0.1
    class Policy:
        def __init__(self,env):
            self.env = env
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            self.obs = tf.placeholder(tf.float32, shape=self.env.observation_space.shape, name="observation")
            self.target = tf.placeholder(tf.float32,shape=self.env.action_space.shape,name="target_action")
            self.action = self.policy_model(self.obs)

        def noise(self,mu=None,sigma=None):
            if mu is None:
                mu = np.zeros(self.env.action_space.shape)
            if sigma is None:
                sigma = 0.01
            return np.random.normal(mu,sigma*np.ones(self.env.action_space.shape))
        def policy_model(self,obs):
            hidden = 64
            with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
                # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
                # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
                dense1 = tf.layers.dense(inputs=tf.expand_dims(obs,0), units=hidden,activation=tf.nn.relu)
                dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu)
                action = tf.layers.dense(inputs=dense2, units= 2, activation=tf.nn.tanh)
                return action

        def predict(self,obs):
            sess = tf.get_default_session()
            return sess.run([self.action],feed_dict={self.obs:obs}) #+ self.noise()

        def update_policy(self,obs,target):
            with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
                sess = tf.get_default_session()
                feed_dict = {self.obs: obs, self.target: target}
                loss, summary = sess.run([self.optimize_policy], feed_dict)
                return summary
        @property
        def optimize_policy(self):
            with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
                loss = tf.reduce_mean((self.action ** 2 - self.target ** 2) ** 0.5)
                policy_step = self.optimizer.minimize(loss)
                tf.summary.scalar('loss', tf.reduce_mean(loss))
                merged = tf.summary.merge_all()
                return policy_step,merged

    class Value:
        def __init__(self,env):
            self.env = env
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            self.obs = tf.placeholder(tf.float32, shape=self.env.observation_space.shape, name="observation")
            self.action = tf.placeholder(tf.float32, shape=np.array([[1],[1]]), name="action")
            self.target = tf.placeholder(tf.float32,shape=[1],name="target_value")
            self.value = self.value_model(self.obs,self.action)

        def value_model(self,obs,action):
            hidden = 64
            with tf.variable_scope("value_fn", reuse=tf.AUTO_REUSE):
                # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
                # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
                dense1 = tf.layers.dense(inputs=tf.expand_dims(obs,0), units=hidden,activation=tf.nn.relu)
                dense1 = tf.concat([dense1,action],axis=1)
                dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu)
                value = tf.layers.dense(inputs=dense2, units=1, activation=tf.nn.tanh)
                return value

        def predict(self,obs,action):
            sess = tf.get_default_session()
            return sess.run([self.value],feed_dict={self.obs:obs,self.action:action})

        def update_value(self,obs,target):
            with tf.variable_scope("value_fn", reuse=tf.AUTO_REUSE):
                sess = tf.get_default_session()
                feed_dict = {self.obs: obs, self.target: target}
                loss, summary = sess.run([self.optimize_value], feed_dict)
                return summary
        @property
        def optimize_value(self):
            with tf.variable_scope("value_fn", reuse=tf.AUTO_REUSE):
                loss = tf.reduce_mean(tf.pow(self.value - self.target,2))
                value_step = self.optimizer.minimize(loss)
                tf.summary.scalar('loss', tf.reduce_mean(loss))
                merged = tf.summary.merge_all()
                return value_step,merged

    def store_transition(self,obs0, action, reward, obs1, terminal1):
        self.memory.append(obs0, action, reward, obs1, terminal1)

    def train(self,):
        sess = tf.get_default_session()
        batch = self.memory.sample(batch_size=64)
        obs = batch[ 'obs0']
        rew = batch['rewards']
        actions = batch['actions']
        advantage = self.critic.predict(obs,actions)
        y = rew+self.gamma*advantage
        self.critic.update_value(obs,y)
        # sess.run(tf.global_variables_initializer())
        # from datetime import datetime
        #
        # now = datetime.now()
        # train_writer = tf.summary.FileWriter('./train/'+ now.strftime("%Y%m%d-%H%M%S") + '/', sess.graph)
        # for i in range(100000):
        #     observation, reward, done, info = self.env.step(self.env.action_space.sample())
        #     loss,summary = self.policy.update_policy(observation,np.array([observation[0]**2]))
        #     print(observation,loss)
        #     train_writer.add_summary(summary, i)
        #     if done:
        #         env.reset()

# env = gym.make('CartPole-v0')
# env.reset()
# new_run = CartPoleAgent(env)
# new_run.train()