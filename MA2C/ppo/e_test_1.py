from policies.gaussian_policy import Actor
from wrappers.critic_wrapper import Critic_Wrapper
from replay_buffer.r_buffer import Buffer
from learning.ppo import PPO

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import random
import tensorflow_probability as tfp

problem = "Hopper-v3"
env = gym.make(problem)
eval_env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

ppo1 = PPO(env, num_states, num_actions, upper_bound, 1500)

# ppo1.load_weights("/home/tony/rl_models/MA2C/ppo/weights/e_test_1_a.ckpt", "/home/tony/rl_models/MA2C/ppo/weights/e_test_1_c.ckpt")

for i in range(1000):
	ppo1.train(1, 1000)
	ppo1.eval_rollout(eval_env)

ppo1.save_weights("/home/tony/rl_models/MA2C/ppo/weights/e_test_1_a.ckpt", "/home/tony/rl_models/MA2C/ppo/weights/e_test_1_c.ckpt")

