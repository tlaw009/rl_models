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

import numpy as np
import robosuite as suite
from gym import spaces
from robosuite import load_controller_config

class Robosuite_Wrapper():

    def __init__(self, domain, task, controller):
        self.config = load_controller_config(default_controller=controller)
        self.env = suite.make(env_name=task, # try with other tasks like "Stack" and "Door"
                            robots=domain,  # try with other robots like "Sawyer" and "Jaco"
                            controller_configs=self.config,
                            has_renderer=False,
                            ignore_done=False,
                            has_offscreen_renderer=False,
                            use_camera_obs=False,
                            reward_shaping=True,
                            )
        self.obs_keys = [key for key, value in self.env.observation_spec().items()]
        
        obs_dim = []
        
        for x in self.obs_keys:
            if x == 'hinge_qpos' or x == 'handle_qpos':
                obs_dim.append(1)
            else:
                obs_dim.append(self.env.observation_spec()[x].shape[0])
                
        self.s_dim = np.sum(obs_dim,dtype=np.int32)
        self.a_dim = self.env.action_dim
        self.a_ub = self.env.action_spec[1][0]
        self.a_lb = self.env.action_spec[0][0]
        
    def env_specs(self):
        return self.s_dim, self.a_dim, self.a_ub, self.a_lb
    
    def step(self, a):
        s, r, d, i = self.env.step(a)

        s = np.concatenate([s[x] for x in self.obs_keys], axis = None)

        return s, r, d, i
    
    def reset(self):
        s = self.env.reset()
        
        s = np.concatenate([s[x] for x in self.obs_keys], axis = None)
        
        return s
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
        