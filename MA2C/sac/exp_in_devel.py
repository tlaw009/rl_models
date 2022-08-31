import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

from learning.sac import SAC

problem = "Humanoid-v3"
env = gym.make(problem)
eval_env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


sac1 = SAC(env, num_states, num_actions, upper_bound, 1000000)
# sac1.load_weights(os.path.dirname(os.path.abspath(__file__))+"/weights")
eval_r = []
for i in range(10000):
	sac1.train(1000)
	sac1.save_weights(os.path.dirname(os.path.abspath(__file__))+"/weights")
	eval_r.append(sac1.eval_rollout(problem))

plt.plot(eval_r)
plt.xlabel("per 1000 steps")
plt.ylabel("Evaluation episodic reward")
plt.show()