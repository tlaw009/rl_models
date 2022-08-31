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

class Critic_Wrapper():
    def __init__(self, state_dim, action_dim):
        self.s_dim=state_dim
        self.a_dim=action_dim
        
    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.s_dim))
        state_input_norm = layers.BatchNormalization()(state_input)
        state_out = layers.Dense(128, activation="relu")(state_input_norm)

        # Action as input
        action_input = layers.Input(shape=(self.a_dim))
        action_out = layers.Dense(128, activation="relu")(action_input)

        # Concatenating
        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(256, activation="relu")(concat)
        outputs = tf.squeeze(layers.Dense(1)(out))

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model