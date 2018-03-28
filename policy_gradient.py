import tensorflow as tf
import gym
import numpy as np
from replay_buffer import Memory
from matplotlib import pyplot as plt

class Value:
    def __init__(self, env,scope):
        self.scope = scope
        self.env = env
        self.action_space = self.env.action_space.shape[0]
        self.obs_space = self.env.observation_space.shape[0]
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            self.obs = tf.placeholder(tf.float32, shape=[None,self.obs_space], name="observation")
            self.action = tf.placeholder(tf.float32, shape=[None,self.action_space], name="action")
            self.target = tf.placeholder(tf.float32, shape=[None,1], name="target_value")
            self.value = self.value_model(self.obs, self.action)
            self.loss = tf.reduce_mean(tf.pow((self.value_model(self.obs,self.action) - self.target),2))
            self.optimize_value = self.optimizer.minimize(self.loss)

    def value_model(self, obs, action):
        hidden = 64
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
            # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
            dense1 = tf.layers.dense(inputs=obs, units=hidden, activation=tf.nn.relu)
            dense1 = tf.concat([dense1, action], axis=1)
            dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu)
            value = tf.layers.dense(inputs=dense2, units=1, activation=tf.nn.tanh)
            return value
    def get_trainable_parameters(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            policy_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            return policy_parameters

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
            grad = tf.gradients(self.value,self.action)
            return sess.run([grad],feed_dict={self.action:action,self.obs:obs})

    def predict(self, obs, action):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            obs = obs.reshape(-1, self.obs_space)
            action = action.reshape(-1,1)
            sess = tf.get_default_session()
            value = sess.run([self.value], feed_dict={self.obs: obs, self.action: action})
            # tf.summary.scalar('value', value)
            return value
    def update_value(self, obs, action, target):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            sess = tf.get_default_session()
            feed_dict = {self.obs: obs, self.target: target, self.action:action}
            loss, _ = sess.run([self.loss,self.optimize_value], feed_dict)

class Policy:
    def __init__(self, env,scope):
        self.scope = scope
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            self.env = env
            self.action_space = self.env.action_space.shape[0]
            self.obs_space = self.env.observation_space.shape[0]
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            self.obs = tf.placeholder(tf.float32, shape=[None, self.obs_space], name="observation")
            self.target = tf.placeholder(tf.float32, shape=[None, self.action_space], name="target_action")
            self.action = self.policy_model(self.obs)

    def noise(self, mu=None, sigma=None):
        if mu is None:
            mu = np.zeros([1,self.action_space])
        if sigma is None:
            sigma = 0.01
        return np.random.normal(mu, sigma * np.ones([1,self.action_space]))

    def policy_model(self, obs):
        hidden = 64
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # obs = tf.placeholder(tf.float32,shape=self.env.observation_space.shape,name="observation")
            # target = tf.placeholder(tf.float32,shape=[1],name="target_action")
            dense1 = tf.layers.dense(inputs=obs, units=hidden, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=hidden, activation=tf.nn.relu)
            action = tf.layers.dense(inputs=dense2, units=1,activation=tf.nn.tanh)
            action = tf.clip_by_value(action,self.env.action_space.low,self.env.action_space.high)
            return action

    def predict(self, obs):
        obs = obs.reshape(-1,self.obs_space)
        sess = tf.get_default_session()
        return sess.run([self.action], feed_dict={self.obs: obs}) + self.noise()

    def update_policy(self, obs, target):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            sess = tf.get_default_session()
            feed_dict = {self.obs: obs, self.target: target}
            loss, summary = sess.run([self.optimize_policy], feed_dict)
            return summary

    def get_trainable_parameters(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            policy_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
            return policy_parameters

    def set_trainable_parameters(self,parameters,tau):
        this_policy_parameters = self.get_trainable_parameters()
        self.soft_assign = [tf.assign(p_t, p*tau+p_t*(tau-1)) for p_t, p in zip(this_policy_parameters,parameters)]
        sess = tf.get_default_session()
        sess.run(self.soft_assign)

    def optimize_policy(self,q_grad,obs):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            obs = obs.reshape(-1, self.obs_space)
            Q_gradient= tf.placeholder(tf.float32,shape=[None,1],name="Q_gradient")
            policy_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.scope)
            policy_grad = tf.gradients(self.action,policy_parameters,Q_gradient)
            # print(policy_parameters)
            # loss = tf.reduce_mean(policy_grad)
            policy_step = self.optimizer.apply_gradients(zip(policy_grad,policy_parameters))
            sess = tf.get_default_session()
            sess.run([policy_step],feed_dict={Q_gradient:q_grad,self.obs:obs})
            # tf.summary.scalar('loss', tf.reduce_mean(loss))
            # merged = tf.summary.merge_all()

gamma = 0.99
episodes = 100
batch_size = 128
max_time_steps = 200
episode_reward = 0
reward_history = []
env = gym.make('MountainCarContinuous-v0')
obs_old = env.reset()
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
agent_policy = Policy(env,"policy")
agent_critic = Value(env,"value")
agent_policy_t = Policy(env,"policy_t")
agent_critic_t = Value(env,"value_t")
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
time = 200
tf.summary.scalar("episode_time_steps", time)
tf.summary.scalar("episode_reward", episode_reward)
merged = tf.summary.merge_all()
with tf.Session() as sess:
    from datetime import datetime
    now = datetime.now()
    train_writer = tf.summary.FileWriter('./train/'+now.strftime("%Y%m%d-%H%M%S")+'/', sess.graph)
    # train_writer = tf.summary.FileWriter('.' + '/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    agent_critic_t.set_trainable_parameters(agent_critic.get_trainable_parameters(), 0)
    agent_policy_t.set_trainable_parameters(agent_policy.get_trainable_parameters(), 0)
    for i in range(episodes):
        print('running episode:',i)
        t = 0
        done = 0
        while not done and t < max_time_steps:
            action = agent_policy.predict(obs_old) #+ agent_policy.noise(0,0.1)
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
            if done or t == max_time_steps-1:
                time = t
                reward_history.append(episode_reward)
                env.reset()
            obs_old = obs
            t += 1
            batch = memory.sample(batch_size)
            obs_batch = batch['obs0']
            action_batch_predict = agent_policy_t.predict(obs_batch)[0]

            value_batch = agent_critic_t.predict(obs_batch,action_batch_predict)[0]
            y = np.array(batch['rewards']) + gamma* np.array(value_batch)#.reshape(-1,batch_size)
            agent_critic.update_value(obs_batch,action_batch_predict,y)
            q_grad = np.array(agent_critic.get_q_gradient(action_batch_predict, obs_batch)).reshape(-1,1)
            agent_policy.optimize_policy(q_grad,obs_batch)
            agent_critic_t.set_trainable_parameters(agent_critic.get_trainable_parameters(), 0.001)
            agent_policy_t.set_trainable_parameters(agent_policy.get_trainable_parameters(), 0.001)
        print(episode_reward)
        summary = sess.run(merged)
        train_writer.add_summary(summary, i)
        time = 200
        episode_reward = 0

# print(reward_history)
# plt.plot(reward_history)