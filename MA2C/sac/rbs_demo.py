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

problem = ("Sawyer", "Door", "JOINT_VELOCITY")
rbs_env = Robosuite_Wrapper("Sawyer", "Door", "JOINT_VELOCITY", True)

num_states, num_actions, upper_bound, lower_bound = rbs_env.env_specs()

sac1 = SAC(rbs_env, num_states, num_actions, upper_bound, 1000000)
sac1.load_weights(os.path.dirname(os.path.abspath(__file__))+"/weights")


for i in range(10):
	sac1.eval_rollout(problem, rbs=True, render=True)