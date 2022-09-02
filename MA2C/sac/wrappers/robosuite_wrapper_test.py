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

from wrappers.robosuite_wrapper import Robosuite_Wrapper

rbs_env = Robosuite_Wrapper("Sawyer", "Door", "JOINT_VELOCITY")

print(rbs_env.env_specs())

print(rbs_env.step(np.ones(8)))