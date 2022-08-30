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

EPSILON = 1e-10

class Actor(Model):

    def __init__(self, action_dimensions, action_bound):
        super().__init__()
        self.action_dim, self.upper_bound = action_dimensions, action_bound
        self.sample_dist = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.action_dim),
                                                                    scale_diag=tf.ones(self.action_dim))
        self.dense1_layer = layers.Dense(256, activation="relu")
        self.dense2_layer = layers.Dense(256, activation="relu")
        self.mean_layer = layers.Dense(self.action_dim)
        self.stdev_layer = layers.Dense(self.action_dim)

    def call(self, state, eval_mode=False):

        a1 = self.dense1_layer(state)
        a2 = self.dense2_layer(a1)
        mu = self.mean_layer(a2)

        log_sigma = self.stdev_layer(a2)
        sigma = tf.exp(log_sigma)
        sigma = tf.clip_by_value(sigma, EPSILON, 2.718)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        
        if eval_mode:
            action_ = mu
        else:
            action_ = tf.math.add(mu, tf.math.multiply(sigma, tf.expand_dims(self.sample_dist.sample(), 0)))
 
        action = tf.tanh(action_)

        log_pi_ = dist.log_prob(action_)     
        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(tf.clip_by_value(1 - action**2, EPSILON, 1.0)), axis=1)
        
        return action*self.upper_bound, log_pi
