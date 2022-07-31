import gym
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import tensorflow_probability as tfp
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx('float64')

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, num_states, num_actions):
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
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    # Takes (s,a,r,s',d) obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    # @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):

        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape1:
            # Get Q value estimates, action used here is from the replay buffer
            q1 = critic_model_1([state_batch, action_batch], training=True)
            # Sample actions from the policy for next states
            pi_a, log_pi_a = actor_model(next_state_batch, eval_mode=True)

            # Get Q value estimates from target Q network
            q1_target = target_critic_1([next_state_batch, pi_a])
            q2_target = target_critic_2([next_state_batch, pi_a])

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.math.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * log_pi_a

            y = tf.stop_gradient(reward_batch + gamma * done_batch * soft_q_target)
            critic1_losses = 0.5 * (
                    tf.losses.MSE(y_true=y, y_pred=q1))
            critic1_loss = tf.nn.compute_average_loss(critic1_losses)

        with tf.GradientTape() as tape2:
            # Get Q value estimates, action used here is from the replay buffer
            q2 = critic_model_2([state_batch, action_batch], training=True)
            # Sample actions from the policy for next states
            pi_a, log_pi_a = actor_model(next_state_batch, eval_mode=True)

            # Get Q value estimates from target Q network
            q1_target = target_critic_1([next_state_batch, pi_a])
            q2_target = target_critic_2([next_state_batch, pi_a])

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.math.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * log_pi_a

            y = tf.stop_gradient(reward_batch + gamma * done_batch * soft_q_target)
            critic2_losses = 0.5 * (
                    tf.losses.MSE(y_true=y, y_pred=q2))
            critic2_loss = tf.nn.compute_average_loss(critic2_losses)

        grads1 = tape1.gradient(critic1_loss, critic_model_1.trainable_variables)
        critic1_optimizer.apply_gradients(zip(grads1,
                                                   critic_model_1.trainable_variables))

        grads2 = tape2.gradient(critic2_loss, critic_model_2.trainable_variables)
        critic2_optimizer.apply_gradients(zip(grads2,
                                                   critic_model_2.trainable_variables))


        with tf.GradientTape() as tape3:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = actor_model(state_batch)

            q1 = critic_model_1([state_batch, pi_a])
            q2 = critic_model_2([state_batch, pi_a])

            soft_q = tf.reduce_mean([q1, q2], axis = 0)

            actor_losses = tf.math.add(tf.math.multiply(alpha,log_pi_a), -soft_q)
            actor_loss = tf.nn.compute_average_loss(actor_losses)

        variables3 = actor_model.trainable_variables
        grads3 = tape3.gradient(actor_loss, variables3,
                                 # unconnected_gradients=tf.UnconnectedGradients.ZERO)
                                    )
        actor_optimizer.apply_gradients(zip(grads3, variables3))

        with tf.GradientTape() as tape4:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = actor_model(state_batch, eval_mode=True)
            alpha_loss = tf.nn.compute_average_loss(-alpha*tf.stop_gradient(log_pi_a +
                                                       target_entropy))

        variables4 = [alpha]
        grads4 = tape4.gradient(alpha_loss, variables4)
        alpha_optimizer.apply_gradients(zip(grads4, variables4))

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
        reward_batch = tf.cast(reward_batch, dtype=tf.float64)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
# @tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
