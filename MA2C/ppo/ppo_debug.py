import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os
import json
import random
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from gym.envs.mujoco.hopper import HopperEnv

tf.keras.backend.set_floatx('float32')

EPSILON = 1e-10

################## GLOBAL SETUP P1 ##################

problem = "Hopper-v2"
env = gym.make(problem)
eval_env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states), flush=True)
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions), flush=True)

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound), flush=True)
print("Min Value of Action ->  {}".format(lower_bound), flush=True)

minibatch_size = 256

##########*****####################*****##########

#################### Auxiliaries ####################

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


##########*****####################*****##########


#################### Replay Buffer ####################

class Buffer:

    def __init__(self, observation_dimensions, action_dimensions, size, gamma=0.99, lam=0.95):

        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, action_dimensions), dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):

        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):

        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        rindex = np.random.choice(self.pointer, minibatch_size)
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer[rindex]),
            np.std(self.advantage_buffer[rindex]),
        )
        return (
            self.observation_buffer[rindex],
            self.action_buffer[rindex],
            (self.advantage_buffer[rindex] - advantage_mean) / advantage_std,
            self.return_buffer[rindex],
            self.logprobability_buffer[rindex],
        )
    def clear(self):
        self.pointer, self.trajectory_start_index = 0, 0

##########*****####################*****##########

#################### Models ####################

class Actor(Model):

    def __init__(self):
        super().__init__()
        self.action_dim = num_actions
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

        covar_m = tf.linalg.diag(sigma**2)

        dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(covar_m))
        if eval_mode:
            action_ = mu
        else:
            action_ = dist.sample()

        action = tf.tanh(action_)

        log_pi_ = dist.log_prob(action_)

        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(tf.clip_by_value(1 - action**2, EPSILON, 1.0)), axis=1)     

        return action*upper_bound, log_pi

def get_critic():
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(256, activation="relu")(state_input)

    out = layers.Dense(256, activation="relu")(state_out)
    outputs = layers.Dense(1, dtype='float32')(out)

    model = tf.keras.Model(state_input, outputs)

    return model

##########*****####################*****##########

#################### GLOBAL SETUP P2 ####################

# Hyperparameters of the PPO algorithm
horizon = 500
iterations = 1000
gamma = 0.99
clip_ratio = 0.2
epochs = 500
lam = 0.97
target_kl = 0.01
beta = 1.0
render = False

actor_model = Actor()
critic_model = get_critic()

lr = 0.0003

policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                                            )
                                                # clipvalue=1.0)

value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                                            )
                                                # clipvalue=1.0)

buffer = Buffer(num_states, num_actions, horizon)


##########*****####################*****##########


#################### Training ####################

observation, episode_return, episode_length = env.reset(), 0, 0
tf_observation = tf.expand_dims(observation, 0)

def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):
    global beta
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        action, log_a = actor_model(observation_buffer)
        print("A: ", tf.reduce_mean(action))
        print("LOG_A: ", tf.reduce_mean(log_a))
        ratio = tf.exp(
            log_a
            - logprobability_buffer
        )
        print("R: ", tf.reduce_mean(ratio), flush=True)
        # c_ratio = None
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        _kl = -beta*tf.math.reduce_max(logprobability_buffer - log_a)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage) + _kl)
        print("LOSS: ", policy_loss)
    policy_grads = tape.gradient(policy_loss, actor_model.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor_model.trainable_variables))
    # print("GRAD: ", policy_grads[0], flush=True)
    action_opt, log_a_opt = actor_model(observation_buffer)
    kl = tf.reduce_mean(
        logprobability_buffer
        - log_a_opt
    )

    if kl < target_kl/1.5:
        beta = beta/2
    if kl > target_kl*1.5:
        beta = beta*2

    return kl

def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic_model(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic_model.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic_model.trainable_variables))


for ite in range(iterations):



    for t in range(horizon):
        if render:
            env.render()


        action, log_pi_a = actor_model(tf_observation)
        action = action[0]

        observation_new, reward, done, _ = env.step(action)

        episode_return += reward
        episode_length += 1


        value_t = critic_model(tf_observation)


        buffer.store(observation, action, reward, value_t, log_pi_a)


        observation = observation_new
        tf_observation = tf.expand_dims(observation, 0)


        terminal = done
        if terminal or (t == horizon - 1):
            last_value = 0 if done else critic_model(tf_observation)
            buffer.finish_trajectory(last_value)
            observation, episode_return, episode_length = env.reset(), 0, 0
            tf_observation = tf.expand_dims(observation, 0)


    for _ in range(epochs):

        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )

        train_value_function(observation_buffer, return_buffer)

    buffer.clear()


##########*****####################*****##########

####################Loss before NaN#####################

# tf.Tensor(0.2181576, shape=(), dtype=float32)
# tf.Tensor(0.1889435, shape=(), dtype=float32)
# tf.Tensor(-0.04418509, shape=(), dtype=float32)
# tf.Tensor(0.0629648, shape=(), dtype=float32)
# tf.Tensor(0.17935935, shape=(), dtype=float32)
# tf.Tensor(-0.04629019, shape=(), dtype=float32)
# tf.Tensor(0.17438054, shape=(), dtype=float32)
# tf.Tensor(0.031371567, shape=(), dtype=float32)
# tf.Tensor(0.020767871, shape=(), dtype=float32)
# tf.Tensor(-0.00432986, shape=(), dtype=float32)
# tf.Tensor(0.1378177, shape=(), dtype=float32)
# tf.Tensor(0.019980386, shape=(), dtype=float32)
# tf.Tensor(0.21751454, shape=(), dtype=float32)
# tf.Tensor(nan, shape=(), dtype=float32)