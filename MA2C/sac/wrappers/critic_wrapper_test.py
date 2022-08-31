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
# import matplotlib.pyplot as plt
import random
import tensorflow_probability as tfp

from wrappers.critic_wrapper import Critic_Wrapper
from policies.gaussian_policy import Actor

problem = "Hopper-v3"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

actor_test = Actor(num_actions, upper_bound)

critic_gen = Critic_Wrapper(num_states, num_actions)
critic_test = critic_gen.get_critic()

obs = env.reset()

tf_obs = tf.expand_dims(obs, 0)
a_test, log_a_test = actor_test(tf_obs)

v_test = critic_test([tf_obs, a_test])

obs_new, _, _, _ = env.step(a_test[0])
tf_obs_new = tf.expand_dims(obs_new, 0)
statex2 = tf.convert_to_tensor([obs, obs_new])
print(statex2.shape)

a_2, loga_2 = actor_test(statex2)

print(a_2.shape, loga_2.shape)

v_2 = critic_test([statex2, a_2])
print(v_2.shape)

critic_test.summary()