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

# real env setup

config = load_controller_config(default_controller="OSC_POSE")

obs_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
                 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object-state',
                 'robot0_proprio-state', 'gripper_to_cube_pos', 'cube_quat', 'cube_pos']


env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=config,
    has_renderer=False,
    ignore_done=False,
    has_offscreen_renderer=False,
    reward_shaping=True,
    use_camera_obs=False,
)


obs_dim = []
for x in obs_keys:
    obs_dim.append(env.observation_spec()[x].shape[0])

state_dim = np.sum(obs_dim, dtype=np.int32)


num_states = state_dim.item()
print("Size of State Space ->  {}".format(num_states), flush=True)
num_actions = env.action_dim
print("Size of Action Space ->  {}".format(num_actions), flush=True)

upper_bound = env.action_spec[1][0]
lower_bound = env.action_spec[0][0]

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

print("State Normalization Initialized", flush=True)

obs_upper = np.zeros(num_states)
obs_lower = np.zeros(num_states)

def obs_norm(state):
    norm_state = np.zeros(num_states)
    for i in range(num_states):
        if state[i] > obs_upper[i]:
            obs_upper[i] =  state[i]
        if state[i] < obs_lower[i]:
            obs_lower[i] = state[i]
        if obs_upper[i] == 0 and obs_lower[i] == 0:
            norm_state[i] = state[i]
        else:
            norm_state[i] = state[i]/(obs_upper[i] - obs_lower[i])
        
    return norm_state
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
    e_greedy = False
    if np.random.rand() < epsilon:
        sampled_actions = np.random.uniform(-1.0, 1.0, num_actions)
        e_greedy = True

    else:
        sampled_actions = tf.squeeze(actor_model(state))

    noise = noise_object()
    # Adding noise to action
    if e_greedy:
        sampled_actions = sampled_actions + noise  

    else:
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
epsilon_INIT = 0.99
epsilon = epsilon_INIT
alpha_INIT = 0.5
alpha = alpha_INIT
eval_flag = False
ep = 0
t_steps = 0
RO_SIZE=1000

while ep < total_episodes:

    if eval_flag:
        prev_state = env.reset()
        prev_state_reshaped = []

        for x in obs_keys:
            prev_state_reshaped.append(prev_state[x])

        prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)
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

            eval_episodic_reward += reward

            state_reshaped = []

            for x in obs_keys:
                state_reshaped.append(state[x])

            state = np.concatenate(np.array(state_reshaped), axis = None)
            state = obs_norm(state)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        eval_ep_reward_list.append(eval_episodic_reward)
        eval_avg_reward = np.mean(eval_ep_reward_list)
        eval_avg_reward_list.append(eval_avg_reward)
        print("EPSILON: ", epsilon, flush=True)
        print("ALPHA: ", alpha, flush=True)
        print("Rollout * {} * eval Reward is ==> {}".format(ep, eval_avg_reward), flush=True)
        ep = ep + 1
        eval_flag = False

    else:
        prev_state = env.reset()

        prev_state_reshaped = []
        for x in obs_keys:
            prev_state_reshaped.append(prev_state[x])

        prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)
        prev_state = obs_norm(prev_state)

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)[0]

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            state_reshaped = []

            for x in obs_keys:
                state_reshaped.append(state[x])

            state = np.concatenate(np.array(state_reshaped), axis = None)
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

        epsilon = np.minimum((1.0 - epsilon)*(best_avg_reward*(1+alpha) - avg_reward)/(best_avg_reward) , epsilon_INIT)
        alpha = alpha_INIT*np.exp((total_episodes - ep)/1000.0)/np.exp(total_episodes/1000.0)
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(eval_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, eval")
plt.show()
##########*****####################*****##########
