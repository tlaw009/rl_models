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

class Critic_Wrapper():
    def __init__(self, state_dim):
        self.s_dim=state_dim
        
    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.s_dim))
        state_out = layers.Dense(256, activation="relu")(state_input)
        # state_out = layers.Dense(32, activation="relu")(state_out)

        out = layers.Dense(256, activation="relu")(state_out)
        outputs = layers.Dense(1, dtype='float64')(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input], outputs)

        return model
