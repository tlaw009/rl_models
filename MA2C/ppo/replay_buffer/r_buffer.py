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

class Buffer:

    def __init__(self, observation_dimensions, action_dimensions, size, minibatch_size=256, gamma=0.99, lam=0.95):

        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, action_dimensions), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.batch_size = minibatch_size
        
        self.buffer_cap = size
        self.pointer = 0
        self.trajectory_start_indices = []
        self.trajectory_start_indices.append(0)

    def store(self, observation, action, reward, logprobability, done):

        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1
        if done and not self.pointer > self.buffer_cap-1:
            self.trajectory_start_indices.append(self.pointer)


    def get(self):
        # Get all data of the buffer
        if self.trajectory_start_indices[-1] == self.buffer_cap-1:
            rindex = np.random.choice(range(len(self.trajectory_start_indices)-1), self.batch_size)
        else:
            rindex = np.random.choice(range(len(self.trajectory_start_indices)), self.batch_size)
        
        isolated_obs=[]
        isolated_a=[]
        isolated_r=[]
        isolated_log_a=[]
        for ri in rindex:
            
            if  ri == len(self.trajectory_start_indices)-1:
                isolated_obs.append(self.observation_buffer[self.trajectory_start_indices[ri]:
                                                       self.buffer_cap])
                isolated_a.append(self.action_buffer[self.trajectory_start_indices[ri]:
                                                       self.buffer_cap])
                isolated_r.append(self.reward_buffer[self.trajectory_start_indices[ri]:
                                                       self.buffer_cap])
                isolated_log_a.append(self.logprobability_buffer[self.trajectory_start_indices[ri]:
                                                       self.buffer_cap])
                
            else:
                isolated_obs.append(self.observation_buffer[self.trajectory_start_indices[ri]:
                                                       self.trajectory_start_indices[ri+1]])
                isolated_a.append(self.action_buffer[self.trajectory_start_indices[ri]:
                                                       self.trajectory_start_indices[ri+1]])
                isolated_r.append(self.reward_buffer[self.trajectory_start_indices[ri]:
                                                       self.trajectory_start_indices[ri+1]])
                isolated_log_a.append(self.logprobability_buffer[self.trajectory_start_indices[ri]:
                                                       self.trajectory_start_indices[ri+1]])

        return (
            isolated_obs,
            isolated_a,
            isolated_r,
            isolated_log_a,
        )
    
    def batch_sample(self, critic_handle):
        s_b, a_b, r_b, l_b = self.get()
        ss_b = []
        as_b = []
        rs_b = []
        ls_b = []
        adv_b = []
        ret_b = []
        sample_idxs = [np.random.choice(range(len(a)-1)) for a in s_b]
        
        for i in range(self.batch_size):
            ss_b.append(s_b[i][sample_idxs[i]])
            as_b.append(a_b[i][sample_idxs[i]])
            rs_b.append(r_b[i][sample_idxs[i]])
            ls_b.append(l_b[i][sample_idxs[i]])
            adv_b.append(self.adv_t(r_b[i][sample_idxs[i]:-1],
                                      critic_handle,
                                      s_b[i][sample_idxs[i]:-1],
                                      s_b[i][sample_idxs[i]+1:]))
            ret_b.append(self.ret_t(r_b[i][sample_idxs[i]:]))
        return (
            tf.convert_to_tensor(ss_b),
            tf.convert_to_tensor(as_b),
            tf.convert_to_tensor(adv_b),
            tf.convert_to_tensor(ret_b),
            tf.convert_to_tensor(ls_b),
            tf.convert_to_tensor(rs_b)
        )
        
    def adv_t(self, r_t, vf, s_t, s_t1):
        ite_gamma_lam = [(self.gamma*self.lam)**i for i in range(len(r_t))]
        delta_ts = r_t + self.gamma*tf.squeeze(vf(s_t1)) - tf.squeeze(vf(s_t))

        return np.sum(np.multiply(ite_gamma_lam, delta_ts))
    
    def ret_t(self, r_t):
        ite_gamma = [self.gamma**i for i in range(len(r_t))]
        
        return np.sum(np.multiply(ite_gamma, r_t))
    
    def clear(self):
        self.pointer = 0
        self.trajectory_start_indices = []
        self.trajectory_start_indices.append(0)