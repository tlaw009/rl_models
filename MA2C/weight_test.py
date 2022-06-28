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
from tensorflow import keras


config = load_controller_config(default_controller="OSC_POSE")

obs_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
                 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object-state',
                 'robot0_proprio-state', 'gripper_to_cube_pos', 'cube_quat', 'cube_pos']

env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=config,
    has_renderer=True,
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
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_dim
print("Size of Action Space ->  {}".format(num_actions))


upper_bound = env.action_spec[1][0]
lower_bound = env.action_spec[0][0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="tanh")(inputs)
    out = layers.Dense(64, activation="tanh")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

total_episodes = 99

trained_actor = get_actor()

trained_actor.load_weights("weights/best_actor.h5")

for ep in range(total_episodes):
    prev_state = env.reset()


    prev_state_reshaped = []

    for x in obs_keys:
        prev_state_reshaped.append(prev_state[x])

    prev_state = np.concatenate(np.array(prev_state_reshaped), axis = None)

    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        sampled_actions = np.squeeze(tf.squeeze(trained_actor(tf_prev_state)))
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(sampled_actions)

        episodic_reward += reward

        state_reshaped = []

        for x in obs_keys:
            state_reshaped.append(state[x])

        state = np.concatenate(np.array(state_reshaped), axis = None)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    print("TOTAL REWARD: ", episodic_reward)