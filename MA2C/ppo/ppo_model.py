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

tf.keras.backend.set_floatx('float64')
# ref: https://github.com/shakti365/soft-actor-critic/blob/master/src/sac.py

EPSILON = 1e-32

################## GLOBAL SETUP P1 ##################

env = HopperEnv()
eval_env = HopperEnv()

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
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, action_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
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
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
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
        rindex = np.random.randint(0, self.pointer - minibatch_size)
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer[rindex:rindex+minibatch_size]),
            np.std(self.advantage_buffer[rindex:rindex+minibatch_size]),
        )
        return (
            self.observation_buffer[rindex:rindex+minibatch_size],
            self.action_buffer[rindex:rindex+minibatch_size],
            (self.advantage_buffer[rindex:rindex+minibatch_size] - advantage_mean) / advantage_std,
            self.return_buffer[rindex:rindex+minibatch_size],
            self.logprobability_buffer[rindex:rindex+minibatch_size],
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
        # Get mean and standard deviation from the policy network
        a1 = self.dense1_layer(state, training=not eval_mode)
        a2 = self.dense2_layer(a1, training=not eval_mode)
        mu = self.mean_layer(a2, training=not eval_mode)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]
        log_sigma = self.stdev_layer(a2, training=not eval_mode)
        sigma = tf.exp(log_sigma)

        covar_m = tf.linalg.diag(sigma**2)

        # dist = tfp.distributions.Normal(mu, sigma)
        dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(covar_m))
        if eval_mode:
            action_ = mu
        else:
            action_ = dist.sample()

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        action = tf.tanh(action_)

        # Calculate the log probability
        log_pi_ = dist.log_prob(action_)

        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper
        log_pi = tf.expand_dims(log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 + EPSILON), axis=1),
                                    -1)        

        return action*upper_bound, log_pi

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(256, activation="relu")(state_input)

    out = layers.Dense(256, activation="relu")(state_out)
    outputs = layers.Dense(1, dtype='float64')(out)

    # Outputs single value for give state-action
    model = tf.keras.Model(state_input, outputs)

    return model

##########*****####################*****##########

#################### GLOBAL SETUP P2 ####################

# Hyperparameters of the PPO algorithm
steps_per_epoch = 2000
epochs = 500
gamma = 0.99
clip_ratio = 0.2
train_iterations = 10
lam = 0.97
target_kl = 0.01

# True if you want to render the environment
render = False


actor_model = Actor()
critic_model = get_critic()


lr = 0.0003

# alpha_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# critic1_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# critic2_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# actor_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")

policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

buffer = Buffer(num_states, num_actions, steps_per_epoch)


# To store reward history of each episode
eval_ep_reward_list = []
eval_avg_reward_list = []

##########*****####################*****##########


#################### Training ####################

observation, episode_return, episode_length = env.reset(), 0, 0
tf_observation = tf.expand_dims(observation, 0)

def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):
    action, log_a = actor_model(observation_buffer)
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            log_a
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        print("R*AB: ", ratio * advantage_buffer)
        print("MA: ", min_advantage)
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor_model.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor_model.trainable_variables))

    action_opt, log_a_opt = actor_model(observation_buffer)
    kl = tf.reduce_mean(
        logprobability_buffer
        - log_a_opt
    )

    kl = tf.reduce_sum(kl)
    return kl

def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic_model(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic_model.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic_model.trainable_variables))

t_steps = 0
RO_SIZE=1000 
RO_index = 0

# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        action, log_pi_a = actor_model(tf_observation)
        action = action[0]
        observation_new, reward, done, _ = env.step(action)
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic_model(tf_observation)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, log_pi_a)

        # Update the observation
        observation = observation_new
        tf_observation = tf.expand_dims(observation, 0)
        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic_model(tf_observation)
            buffer.finish_trajectory(last_value)
            observation, episode_return, episode_length = env.reset(), 0, 0
            tf_observation = tf.expand_dims(observation, 0)

    # Get values from the buffer
    # (
    #     observation_buffer,
    #     action_buffer,
    #     advantage_buffer,
    #     return_buffer,
    #     logprobability_buffer,
    # ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_iterations):

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

        if kl > 1.5 * target_kl:
            # Early Stopping
            break
    buffer.clear()
    # Update the value function


    # Print mean return and length for each epoch
    # print(
    #     f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    # )

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(eval_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, train")
plt.show()

##########*****####################*****##########
