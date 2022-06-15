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

# demo env setup

# demo_path = "/home/tony/rl_demo/1653334277_0912106"

# hdf5_path = os.path.join(demo_path, "demo.hdf5")

# f = h5py.File(hdf5_path, "r")

# env_name = f["data"].attrs["env"]

# env_info = json.loads(f["data"].attrs["env_info"])

# demo_env = suite.make(
#     **env_info,
#     has_renderer=False,
#     has_offscreen_renderer=False,
#     ignore_done=True,
#     use_camera_obs=False,
#     reward_shaping=True,
#     control_freq=20,
# )

# demos = list(f["data"].keys())

# real env setup

config = load_controller_config(default_controller="OSC_POSE")

obs_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
                 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object-state',
                 'robot0_proprio-state', 'gripper_to_cube_pos', 'cube_quat', 'cube_pos']

# problem = "Pendulum-v1"
# env = gym.make(problem)


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
        # print("REWARD_BATCH: ", reward_batch)

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            # print("TARGET_CRITIC: ", tf.math.reduce_mean(target_critic(
            #     [next_state_batch, target_actions], training=True
            # )))
            # print("CRITIC_VALUE: ", tf.math.reduce_mean(critic_model(
            #     [next_state_batch, target_actions], training=True
            # )))
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # print("CRITIC_VALUE: ", critic_value)
        # print("CRITIC_LOSS: ", critic_loss)
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # print("CRITIC_GRADIENT: ", critic_grad)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # critic_value = reward_batch
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            # actor_loss = critic_value

        # print("ACTOR_LOSS: ", actor_loss)
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
    # out = layers.Flatten()(inputs)
    out = layers.Dense(64, activation="tanh")(inputs)
    out = layers.Dense(64, activation="tanh")(out)
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
    state_out = layers.Dense(16, activation="tanh")(state_input)
    state_out = layers.Dense(32, activation="tanh")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="tanh")(action_input)


    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64, activation="tanh")(concat)
    out = layers.Dense(64, activation="tanh")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model



def policy(state, noise_object):
    # print(state.shape)
    e_greedy = False
    if np.random.rand() < epsilon:
        sampled_actions = np.random.uniform(-1.0, 1.0, num_actions)
        e_greedy = True
        # print("RANDOM SAMPLED")
    else:
        sampled_actions = tf.squeeze(actor_model(state))
        # print("ACTOR MODEL")
    # print(sampled_actions.shape)
    noise = noise_object()
    # Adding noise to action
    if e_greedy:
        sampled_actions = sampled_actions + noise  
        # sampled_actions = sampled_actions 
    else:
        sampled_actions = sampled_actions.numpy() + noise  
        # sampled_actions = sampled_actions.numpy()

    # print("SAMPLED_ACTIONS: ", sampled_actions)

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    # print("BOUNDS: ", lower_bound, upper_bound)
    # print("LEGAL_ACTION:", legal_action)

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
critic_lr = 0.003
actor_lr = 0.002

critic_optimizer = tf.keras.optimizers.SGD(learning_rate=critic_lr, momentum=0.05, nesterov=False, name="SGD")
actor_optimizer = tf.keras.optimizers.SGD(learning_rate=actor_lr, momentum=0.05, nesterov=False, name="SGD")

total_episodes = 4000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.05

buffer = Buffer(100000, 64)

# populate buffer with demo
# demo_sample_count = 0

# while demo_sample_count < buffer.buffer_capacity/5.0:
#     print("Playing back random episode... (press ESC to quit)")

#     # # select an episode randomly
#     demo_ep = random.choice(demos)

#     # read the model xml, using the metadata stored in the attribute for this episode
#     model_xml = f["data/{}".format(demo_ep)].attrs["model_file"]

#     demo_env.reset()
#     xml = postprocess_model_xml(model_xml)
#     demo_env.reset_from_xml_string(xml)
#     demo_env.sim.reset()
#     # env.viewer.set_camera(0)

#     # load the flattened mujoco states
#     demo_states = f["data/{}/states".format(demo_ep)][()]


#     # load the initial state
#     demo_env.sim.set_state_from_flattened(demo_states[0])
#     demo_prev_state = None
#     demo_env.sim.forward()

#     # load the actions and play them back open-loop
#     demo_actions = np.array(f["data/{}/actions".format(demo_ep)][()])
#     demo_num_actions = demo_actions.shape[0]

#     for j, action in enumerate(demo_actions):
#         demo_state, demo_reward, demo_done, demo_info = demo_env.step(action)
#         # print("DEMO ACTION, ", action)
#         # print("DEMO REWARD, ", demo_reward)
#         # demo_env.render()
#         demo_state_reshaped = []

#         for x in obs_keys:
#             demo_state_reshaped.append(demo_state[x])

#         demo_state = np.concatenate(np.array(demo_state_reshaped), axis = None)
#         # print("STATE DIM, ", num_states)
#         if not j == 0:
#             buffer.record((demo_prev_state, action, demo_reward, demo_state))
#             demo_sample_count += 1
#             print("DEMO SAMPLE LOADED: ", demo_sample_count)

#         demo_prev_state = demo_state

#         if j < demo_num_actions - 1:
#             # ensure that the actions deterministically lead to the same recorded states
#             state_playback = demo_env.sim.get_state().flatten()
#             if not np.all(np.equal(demo_states[j + 1], state_playback)):
#                 err = np.linalg.norm(demo_states[j + 1] - state_playback)
#                 print(f"[warning] playback diverged by {err:.2f} for ep {demo_ep} at step {j}")

# f.close()

# To store reward history of each episode
eval_ep_reward_list = []
eval_avg_reward_list = []
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

##########*****####################*****##########


#################### Training ####################

best_avg_reward = 0.0
epsilon_INIT = 0.99
epsilon = epsilon_INIT
alpha_INIT = 0.5
alpha = alpha_INIT
eval_flag = False
ep = 0
while ep < total_episodes:

    if eval_flag:
        prev_state = env.reset()
        prev_state_reshaped = []

        for x in obs_keys:
            prev_state_reshaped.append(prev_state[x])

        prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)

        eval_episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            sampled_actions = np.squeeze(tf.squeeze(actor_model(tf_prev_state)))
            # print("ACTION: ", action)
            # print("BUFFER AVG REWARD: ", avg_reward)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(sampled_actions)

            eval_episodic_reward += reward
            # print("ACT/R: ", action, "/", reward)

            # reward = reward + (prev_dist_to_goal - np.linalg.norm(state['gripper_to_cube_pos']))
            # prev_dist_to_goal = np.linalg.norm(state['gripper_to_cube_pos'])

            # print(reward)
            state_reshaped = []

            for x in obs_keys:
                state_reshaped.append(state[x])

            state = np.concatenate(np.array(state_reshaped), axis = None)

            # if ep > total_episodes / 6.0:
            # buffer.record((prev_state, action, reward, state))

            # episodic_reward += reward

            # print(epsilon)

            # buffer.learn()
            # update_target(target_actor.variables, actor_model.variables, tau)
            # update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state
            # step_index = step_index + 1


        # print("TOTAL REWARD: ", eval_episodic_reward)
        eval_ep_reward_list.append(eval_episodic_reward)
        eval_avg_reward = np.mean(eval_ep_reward_list[-40:])
        eval_avg_reward_list.append(eval_avg_reward)

        print("Episode * {} * Avg eval Reward is ==> {}".format(ep, eval_avg_reward), flush=True)
        ep = ep + 1
        eval_flag = False

    else:
    # step_index = 1
    # step_max = 5000
    # avg_reward = np.mean(buffer.reward_buffer)
        prev_state = env.reset()

        # prev_dist_to_goal = np.linalg.norm(prev_state['gripper_to_cube_pos'])

        prev_state_reshaped = []
        for x in obs_keys:
            prev_state_reshaped.append(prev_state[x])

        prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)

        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)[0]
            # print("ACTION: ", action)
            # print("BUFFER AVG REWARD: ", avg_reward)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
            # print("ACT/R: ", action, "/", reward)

            # reward = reward + (prev_dist_to_goal - np.linalg.norm(state['gripper_to_cube_pos']))
            # prev_dist_to_goal = np.linalg.norm(state['gripper_to_cube_pos'])

            # print(reward)
            state_reshaped = []

            for x in obs_keys:
                state_reshaped.append(state[x])

            state = np.concatenate(np.array(state_reshaped), axis = None)

            # if ep > total_episodes / 6.0:
            buffer.record((prev_state, action, reward, state))

            episodic_reward += reward

            # print(epsilon)

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state
            # step_index = step_index + 1

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward), flush=True)
        if(avg_reward > best_avg_reward):        
            actor_model.save_weights("weights/best_actor.h5")
            critic_model.save_weights("weights/best_critic.h5")

            target_actor.save_weights("weights/best_target_actor.h5")
            target_critic.save_weights("weights/best_target_critic.h5")
            best_avg_reward = avg_reward
        avg_reward_list.append(avg_reward)

        print("EPSILON: ", epsilon, flush=True)
        print("ALPHA: ", alpha, flush=True)
        epsilon = epsilon_INIT*(best_avg_reward*(1+alpha) - avg_reward)/(best_avg_reward* (1+alpha))
        alpha = alpha_INIT*np.exp((total_episodes - ep)/1000.0)/np.exp(total_episodes/1000.0)
        eval_flag = True
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, train")
plt.show()
plt.plot(eval_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward, eval")
plt.show()
##########*****####################*****##########

actor_model.save_weights("weights/custom_actor_final.h5")
critic_model.save_weights("weights/custom_critic_final.h5")

target_actor.save_weights("weights/custom_target_actor_final.h5")
target_critic.save_weights("weights/custom_target_critic_final.h5")