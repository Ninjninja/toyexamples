import gym
from cartpole_agent import CartPoleAgent

env = gym.Env()
agent = CartPoleAgent(env)
env.s