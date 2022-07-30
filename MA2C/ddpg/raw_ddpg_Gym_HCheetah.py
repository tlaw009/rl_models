import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
from robosuite import load_controller_config
import h5py
from robosuite.utils.mjcf_utils import postprocess_model_xml
import os
import json
import random

################## GLOBAL SETUP P1 ##################
EPSILON = 1e-16

problem = "HalfCheetah-v2"
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


##########*****####################*****##########

#################### Auxiliaries ####################

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


###########################
#Observation normalization#
###########################

running_shift = np.zeros(num_states)
running_scale = np.ones(num_states)
running_momentum = 0.9
init_period = 256 # should be consistent with batch size
var_batch = np.zeros([init_period, num_states])
sample_index = 0

def obs_norm(state):
    global sample_index

    norm_state = np.zeros(num_states)
    var_batch[sample_index] = state
    sample_index = (sample_index + 1)%init_period

    for i in range(num_states):
        running_shift[i] = running_shift[i]* running_momentum + state[i]* (1-running_momentum)
        if len(var_batch) >= init_period:
            if not state[i] == running_shift[i]:
                running_scale[i] = running_scale[i]* running_momentum + np.var(var_batch, axis=0)[i]* (1-running_momentum)

        norm_state[i] = (state[i]-running_shift[i])/np.sqrt(running_scale[i]+EPSILON)
        
    return norm_state

print("State Normalization Initialized", flush=True)

##########*****####################*****##########


#################### Replay Buffer ####################

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions]
            )
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value), axis = 0)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)

        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value, axis = 0)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
# @tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

##########*****####################*****##########

#################### Models ####################

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

##################################
#Modified for TD advantage critic#
##################################

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation="relu")(action_input)


    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model



def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))

    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise  

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

##########*****####################*****##########

#################### GLOBAL SETUP P2 ####################

std_dev = 0.01
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.0003
actor_lr = 0.0003

# critic_optimizer = tf.keras.optimizers.SGD(learning_rate=critic_lr, momentum=0.05, nesterov=False, name="SGD")
# actor_optimizer = tf.keras.optimizers.SGD(learning_rate=actor_lr, momentum=0.05, nesterov=False, name="SGD")

critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

total_episodes = 1000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

BATCH_SIZE = 256
buffer = Buffer(100000, BATCH_SIZE)

# To store reward history of each episode
eval_ep_reward_list = []
eval_avg_reward_list = []
##########*****####################*****##########

#################### Training ####################
eval_flag = False
ep = 0
t_steps = 0
RO_SIZE=1000

while ep < total_episodes:

    if eval_flag:
        prev_state = env.reset()
        prev_state = obs_norm(prev_state)

        eval_episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            sampled_actions = np.squeeze(tf.squeeze(actor_model(tf_prev_state)))

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(sampled_actions)
            state = obs_norm(state)

            eval_episodic_reward += reward

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        eval_ep_reward_list.append(eval_episodic_reward)
        eval_avg_reward = np.mean(eval_ep_reward_list)
        eval_avg_reward_list.append(eval_avg_reward)

        print("Rollout * {} * eval Reward is ==> {}".format(ep, eval_episodic_reward), flush=True)
        print("Rollout * {} * eval Avg Reward is ==> {}".format(ep, eval_avg_reward), flush=True)
        ep = ep + 1
        eval_flag = False

    else:
        prev_state = env.reset()
        prev_state = obs_norm(prev_state)

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)[0]

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
            state = obs_norm(state)

            buffer.record((prev_state, action, reward, state))

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            t_steps += 1
            if t_steps%RO_SIZE == 0:
                eval_flag = True
            # End this episode when `done` is True
            if done:
                break

            prev_state = state

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(eval_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, eval")
plt.show()
##########*****####################*****##########