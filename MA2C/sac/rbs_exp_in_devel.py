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
from wrappers.robosuite_wrapper import Robosuite_Wrapper

rbs_env = Robosuite_Wrapper("Sawyer", "Door", "JOINT_VELOCITY")

num_states, num_actions, upper_bound, lower_bound = rbs_env.env_specs()

sac1 = SAC(rbs_env, num_states, num_actions, upper_bound, 1000000)

eval_r = []
for i in range(3000):
	sac1.train(1000)
	sac1.save_weights(os.path.dirname(os.path.abspath(__file__))+"/weights")
	eval_r.append(sac1.eval_rollout(problem))

plt.plot(eval_r)
plt.xlabel("per 1000 steps")
plt.ylabel("Evaluation episodic reward")
plt.show()