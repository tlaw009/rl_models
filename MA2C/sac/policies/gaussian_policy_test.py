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

from policies.gaussian_policy import Actor

problem = "Hopper-v3"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

actor_test = Actor(num_actions, upper_bound)

obs = env.reset()
tf_obs = tf.expand_dims(obs, 0)

a_test, log_a_test = actor_test(tf_obs)
print(a_test, log_a_test)

actor_test.summary()