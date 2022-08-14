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
# ref: https://github.com/shakti365/soft-actor-critic/blob/master/src/sac.py

EPSILON = 1e-10

################## GLOBAL SETUP P1 ##################
rand_seed = 1929
problem = "Humanoid-v2"
env = gym.make(problem)
eval_env = gym.make(problem)

env.seed(rand_seed)
eval_env.seed(rand_seed)

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
NaN_found = False
train_log = []
log_len = 24
log_index = 0

def training_log(data):
    global NaN_found, train_log, log_len, log_index

    if not NaN_found:
        if len(train_log) == log_len:
            train_log[log_index] = data

        if len(train_log) < log_len:
            train_log.append(data)
            
        log_index = (log_index+1)%log_len
        if True in tf.math.is_nan(data):
            NaN_found = True
            for x in train_log:
                print("*********************", flush=True)
                print("A: ", x[0], flush=True)
                print("LOG_A: ", x[1], flush=True)
                print("POLICY_LOSS: ", x[2], flush=True)
                print("FIRST_LAYER_GRAD: ", x[3], flush=True)
                print("ALPHA: ", x[4], flush=True)
                print("SOFT_Q: ", x[5], flush=True)
                print("*********************", flush=True)

###########################
#Observation normalization#
###########################

obs_upper = np.zeros(num_states)
obs_lower = np.zeros(num_states)

def obs_norm(state_batch):
    if len(state_batch.shape) == 2:
        norm_state_batch = []
        for state in state_batch:
            norm_state = np.zeros(num_states)
            for i in range(num_states):
                if state[i] > obs_upper[i]:
                    obs_upper[i] = state[i]
                if state[i] < obs_lower[i]:
                    obs_lower[i] = state[i]
                norm_state[i] = state[i]/(obs_upper[i] - obs_lower[i]+EPSILON)
            norm_state_batch.append(norm_state)

        return norm_state_batch
    else: 
        state = state_batch
        norm_state = np.zeros(num_states)
        for i in range(num_states):
            if state[i] > obs_upper[i]:
                obs_upper[i] = state[i]
            if state[i] < obs_lower[i]:
                obs_lower[i] = state[i]
            norm_state[i] = state[i]/(obs_upper[i] - obs_lower[i]+EPSILON)

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
            q1 = critic_model_1([state_batch, action_batch])
            # Sample actions from the policy for next states
            pi_a, log_pi_a = actor_model(next_state_batch)

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
            q2 = critic_model_2([state_batch, action_batch])
            # Sample actions from the policy for next states
            pi_a, log_pi_a = actor_model(next_state_batch)

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

        training_log([tf.reduce_mean(pi_a), tf.reduce_mean(log_pi_a), tf.reduce_mean(actor_loss)
            , tf.reduce_mean(grads3[0]),tf.reduce_mean(alpha), tf.reduce_mean(soft_q)])

        with tf.GradientTape() as tape4:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = actor_model(state_batch)
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
        state_batch = tf.convert_to_tensor(
                                            obs_norm(self.state_buffer[batch_indices]))
                                            # self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float64)
        next_state_batch = tf.convert_to_tensor(
                                            obs_norm(self.next_state_buffer[batch_indices]))
                                            # self.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
# @tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

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
        a1 = self.dense1_layer(state)
        a2 = self.dense2_layer(a1)
        mu = self.mean_layer(a2)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]
        log_sigma = self.stdev_layer(a2)
        sigma = tf.exp(log_sigma)

        sigma = tf.clip_by_value(sigma, EPSILON, 2.718)

        # covar_m = tf.linalg.diag(sigma**2)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

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
        log_pi = tf.expand_dims(log_pi_ - tf.reduce_sum(tf.math.log(tf.clip_by_value(1 - action**2, EPSILON, 1.0)), axis=1),
                                    -1)        

        return action*upper_bound, log_pi

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)
    # state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation="relu")(action_input)

    # Concatenating
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    outputs = layers.Dense(1, dtype='float64')(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

##########*****####################*****##########

#################### GLOBAL SETUP P2 ####################

actor_model = Actor()
critic_model_1 = get_critic()
critic_model_2 = get_critic()
target_critic_1 = get_critic()
target_critic_2 = get_critic()

alpha=tf.Variable(0.0, dtype=tf.float64)
target_entropy = -np.prod(num_actions)

# Making the weights equal initially
target_critic_1.set_weights(critic_model_1.get_weights())
target_critic_2.set_weights(critic_model_2.get_weights())


lr = 0.0003

# alpha_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# critic1_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# critic2_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")
# actor_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.05, nesterov=False, name="SGD")

alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

total_episodes = 5000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
BATCH_SIZE = 256
buffer = Buffer(1000000, BATCH_SIZE)


# To store reward history of each episode
eval_ep_reward_list = []
eval_avg_reward_list = []

##########*****####################*****##########


#################### Training ####################

t_steps = 0
RO_SIZE=1000 
RO_index = 0

while t_steps < 10000000:

    prev_state = env.reset()

    episodic_reward = 0

    while True:
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(
                                                            obs_norm(prev_state)), 0)
                                                            # prev_state), 0)

        action, log_a = actor_model(tf_prev_state)

        action = action[0]

        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        if done:
            end = 0
        else:
            end = 1

        buffer.record((prev_state, action, reward, state, end))

        episodic_reward += reward

        buffer.learn()

        update_target(target_critic_1.variables, critic_model_1.variables, tau)
        update_target(target_critic_2.variables, critic_model_2.variables, tau)
        t_steps += 1

        if t_steps%RO_SIZE == 0:

            eval_prev_state = eval_env.reset()

            eval_ep_reward = 0

            while True:
                # eval_env.render()

                eval_tf_prev_state = tf.expand_dims(tf.convert_to_tensor(
                                                                        obs_norm(eval_prev_state)), 0)
                                                                        # eval_prev_state), 0)

                eval_action, eval_log_a = actor_model(eval_tf_prev_state, eval_mode=True)

                eval_action = eval_action[0]

                # Recieve state and reward from environment.
                eval_state, eval_reward, eval_done, info = eval_env.step(eval_action)

                eval_ep_reward += eval_reward

                if eval_done:
                    break

                eval_prev_state = eval_state

            eval_ep_reward_list.append(eval_ep_reward)
            eval_avg_reward = np.mean(eval_ep_reward_list)
            print("Rollout * {} * Avg Reward is ==> {}".format(RO_index, eval_avg_reward), flush=True)
            print("Rollout * {} * Epsiodic Reward is ==> {}".format(RO_index, eval_ep_reward), flush=True)
            print("TOTAL STEPS: ", t_steps, flush=True)
            eval_avg_reward_list.append(eval_avg_reward)
            RO_index += 1 

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(eval_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, train")
plt.show()

##########*****####################*****##########

# actor_model.save_weights("weights/custom_sac_actor_final.h5")
# critic_model.save_weights("weights/custom_sac_critic_final.h5")

# target_actor.save_weights("weights/custom_sac_target_actor_final.h5")
# target_critic.save_weights("weights/custom_sac_target_critic_final.h5")
