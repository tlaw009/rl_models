import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from wrappers.critic_wrapper import Critic_Wrapper
from policies.gaussian_policy import Actor
from replay_buffer.r_buffer_sac import Buffer

problem = "Hopper-v3"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

actor_test = Actor(num_actions, upper_bound)

buffer1 = Buffer(num_states, num_actions, 100, 10)

prev_obs = env.reset()

for i in range(buffer1.buffer_capacity):
    a, _ = actor_test(tf.expand_dims(prev_obs, 0))
    obs, r, d, _ = env.step(a[0])
    
    buffer1.record((prev_obs, a[0], r, obs, d))
    
    prev_obs = obs
    
print(buffer1.sample())