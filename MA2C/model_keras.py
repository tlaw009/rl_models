import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import sys

import random

import argparse

import logging
import datetime
from os.path import dirname, join, abspath
import time
# import metaworld


#################### Auxiliaries ####################

# def make_env(name):
#     """Create an environment from metaworld env lists"""
#     ml1 = metaworld.ML1(name)
#     env = ml1.train_classes[name]()  # Create an environment with task `pick_place`
#     task = random.choice(ml1.train_tasks)
#     env.set_task(task)  # Set task

#     return env



class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.15, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

##########*****####################*****##########



#################### Training ####################

class Buffer:
    def __init__(self, s_d, a_d, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, s_d))
        self.action_buffer = np.zeros((self.buffer_capacity, a_d))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, s_d))

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


    @tf.function
    def update(self, ta_handle, tc_handle, am_handle, cm_handle, state_batch, action_batch, reward_batch, next_state_batch, gamma, critic_optimizer, actor_optimizer):
        with tf.GradientTape() as tape:
            target_actions = ta_handle(next_state_batch, training=True)
            # print(tc_handle(
            #     next_state_batch, training=True
            # ))
            y = reward_batch + gamma * tc_handle(
                next_state_batch, training=True
            )

            print("Y", y)
            critic_value = cm_handle(state_batch, training=True)

            # print("CRITIC VALUE: ")
            # tf.print(critic_value, output_stream=sys.stderr)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            # print("CRITIC LOSS: ")
            # tf.print(critic_loss, output_stream=sys.stderr)


        critic_grad = tape.gradient(critic_loss, cm_handle.trainable_variables)
        # print("CRITIC GRADIENT: ", critic_grad)
        critic_optimizer.apply_gradients(
            zip(critic_grad, cm_handle.trainable_variables)
        )
        with tf.GradientTape() as tape:
            actions = am_handle(state_batch, training=True)
            critic_value = cm_handle(state_batch, training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            # print(actor_loss)

        actor_grad = tape.gradient(actor_loss, am_handle.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, am_handle.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, ta_handle, tc_handle, am_handle, cm_handle, gamma, critic_optimizer, actor_optimizer):
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
        self.update(ta_handle, tc_handle, am_handle, cm_handle, state_batch, action_batch, reward_batch, next_state_batch, gamma, critic_optimizer, actor_optimizer)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def training(env_name, a_lr, c_lr, max_epoch, gamma, tau):

    #-------------------define hyper parameters-------------------#
    # env = make_env(env_name)
    env = gym.make("Pendulum-v1")
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]))

    actor_model = get_actor(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high_repr))
    critic_model = get_critic(env.observation_space.shape[0])
    
    target_actor = get_actor(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high_repr))
    target_critic = get_critic(env.observation_space.shape[0])

    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_lr = c_lr
    actor_lr = a_lr

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = max_epoch
    gamma = gamma
    tau = tau

    buffer = Buffer(env.observation_space.shape[0], env.action_space.shape[0], 50000, 64)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    #-------------------define hyper parameters-------------------#

    #-------------------training in episodes-------------------#
    for ep in range(total_episodes):
        prev_state = env.reset()
        episodic_reward = 0

        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            actions = policy(tf_prev_state, actor_model, ou_noise, float(env.action_space.low_repr), float(env.action_space.high_repr))

            state, reward, done, info = env.step(actions)
            # print(reward)
            buffer.record((prev_state, actions, reward, state))

            episodic_reward += reward

            buffer.learn(ta_handle=target_actor, tc_handle=target_critic, am_handle=actor_model, cm_handle=critic_model, gamma=gamma, critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
    #-------------------training in episodes-------------------#

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

##########*****####################*****##########

#################### Models ####################

def get_actor(s_d, a_d, action_bound):
    # Initialize weights between -3e-3 and 3-e3
    

    # print("DEBUG")
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    # model = tf.keras.models.Sequential()

    inputs = layers.Input(shape=(s_d,))
    # print("DEBUG")
    # model.add(layers.Input(shape=(s_d,)))

    out = layers.Dense(64, activation="relu")(inputs)
    # print("DEBUG")
    # model.add(layers.Dense(64, activation="relu"))

    # print("DEBUG")
    out = layers.Dense(64, activation="relu")(out)
    # model.add(layers.Dense(64, activation="relu"))

    outputs = layers.Dense(a_d, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * action_bound
    model = tf.keras.Model(inputs, outputs)

    return model


##################################
#Modified for TD advantage critic#
##################################
def get_critic(s_d):
    # State as input
    state_input = layers.Input(shape=(s_d))
    state_out = layers.Dense(64, activation="relu")(state_input)

    out = layers.Dense(64, activation="relu")(state_out)

    outputs = layers.Dense(1)(out)

    # Outputs single value for given state
    model = tf.keras.Model(state_input, outputs)

    return model

# define Policy for sampling
def policy(state, am_handle, noise_object, action_bound_low, action_bound_high):
    sampled_actions = tf.squeeze(am_handle(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, action_bound_low, action_bound_high)

    return [np.squeeze(legal_action)]



#################### Main ####################
def main(args):

    training("pick-place-v2", args['actor_lr'], args['critic_lr'], args['max_epoch'], args['gamma'], args['tau'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='provide arguments for custom agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    # parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    # parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    # parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    # parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max_epoch', help='max num of episodes to do while training', default=50000)
    # parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    # parser.add_argument('--render-env', help='render the gym env', action='store_true')
    # parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    # parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    # parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')  
    args = vars(parser.parse_args())
    main(args)