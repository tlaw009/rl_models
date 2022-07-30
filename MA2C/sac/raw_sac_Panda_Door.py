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
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.mjcf_utils import postprocess_model_xml


tf.keras.backend.set_floatx('float64')
# ref: https://github.com/shakti365/soft-actor-critic/blob/master/src/sac.py

EPSILON = 1e-16

################## GLOBAL SETUP P1 ##################

config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=config,
    has_renderer=False,
    ignore_done=False,
    has_offscreen_renderer=False,
    reward_shaping=True,
    use_camera_obs=False,
)

eval_env = suite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=config,
    has_renderer=False,
    ignore_done=False,
    has_offscreen_renderer=False,
    reward_shaping=True,
    use_camera_obs=False,
)

obs_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 
            'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 
            'robot0_gripper_qvel', 'door_pos', 'handle_pos', 
            'door_to_eef_pos', 'handle_to_eef_pos', 'hinge_qpos', 'handle_qpos', 
            'robot0_proprio-state', 'object-state']

obs_dim = []
for x in obs_keys:
    if x == 'hinge_qpos' or x == 'handle_qpos':
        obs_dim.append(1)
    else:
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

###########################
#Observation normalization#
###########################

running_shift = np.zeros(num_states)
running_scale = np.ones(num_states)
running_momentum = 0.99
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

        covar_m = tf.linalg.diag(sigma**2)

        # dist = tfp.distributions.Normal(mu, sigma)
        dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(covar_m))
        if eval_mode:
            action = tf.tanh(mu)
            log_pi = tf.constant(0, dtype='float64')
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

alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

total_episodes = 5000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
BATCH_SIZE = 256
buffer = Buffer(100000, BATCH_SIZE)


# To store reward history of each episode
eval_ep_reward_list = []
eval_avg_reward_list = []
ep_reward_list = []
avg_reward_list = []

##########*****####################*****##########


#################### Training ####################

ep = 0
t_steps = 0
RO_SIZE=1000 
RO_index = 0

while t_steps < 1000000:

    prev_state = env.reset()

    prev_state_reshaped = []
    for x in obs_keys:
        prev_state_reshaped.append(prev_state[x])

    prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)
    prev_state = obs_norm(prev_state)

    episodic_reward = 0

    while True:
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action, log_a = actor_model(tf_prev_state)

        action = action[0]

        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        state_reshaped = []

        for x in obs_keys:
            state_reshaped.append(state[x])

        state = np.concatenate(np.array(state_reshaped), axis = None)
        state = obs_norm(state)

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

            eval_prev_state_reshaped = []
            for x in obs_keys:
                eval_prev_state_reshaped.append(eval_prev_state[x])

            eval_prev_state = np.concatenate(np.array(eval_prev_state_reshaped), axis = None)
            eval_prev_state = obs_norm(eval_prev_state)

            eval_ep_reward = 0

            while True:
                # eval_env.render()

                eval_tf_prev_state = tf.expand_dims(tf.convert_to_tensor(eval_prev_state), 0)

                eval_action, eval_log_a = actor_model(eval_tf_prev_state, eval_mode=True)

                eval_action = eval_action[0]

                # Recieve state and reward from environment.
                eval_state, eval_reward, eval_done, info = eval_env.step(eval_action)

                eval_state_reshaped = []

                for x in obs_keys:
                    eval_state_reshaped.append(eval_state[x])

                eval_state = np.concatenate(np.array(eval_state_reshaped), axis = None)
                eval_state = obs_norm(eval_state)

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

    ep_reward_list.append(episodic_reward)

    ep = ep + 1

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
