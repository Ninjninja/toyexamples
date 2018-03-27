from cartpole_agent import CartPoleAgent
import gym
import tensorflow as tf
from replay_buffer import Memory

episodes = 1000
max_time_steps = 200
env = gym.make('CartPole-v0')
obs_old = env.reset()
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
agent = CartPoleAgent(env, memory)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(episodes):
        t = 0
        done = 0
        while not done and t < max_time_steps:
            action = agent.policy.predict(obs_old)
            obs, rew, done, info = env.step(action)
            agent.store_transition(obs_old, action, rew, obs, done)
            obs_old = obs
            t += 1
