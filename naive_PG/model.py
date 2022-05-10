import tensorflow as tf
import numpy as np
import random
import gym

import tflearn

import logging
import datetime
from os.path import dirname, join, abspath
import time
import metaworld

## ref: https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py


class ActorNetwork(object):
    def init(self, env, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]


        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]
        
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        self.unnormalized_actor_gradients = #implement proper gradient#

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):

        inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, self.s_dim))
        # net = tflearn.fully_connected(inputs, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = self.dense_1(inputs)
        net = self.dense_2(net)
        out = self.logits(net)
        # net = tflearn.activations.relu(net)
        # net = tflearn.fully_connected(net, 300)
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(
        #     net, self.a_dim, activation='tanh', weights_init=w_init)
        # # Scale output to -action_bound to action_bound
        # scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out


def make_env(name):
    """Create an environment."""
    ml1 = metaworld.ML1(name)
    env = ml1.train_classes[name]()  # Create an environment with task `pick_place`
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task

    obs = env.reset() 

    return env, obs

def learning(env_name, policy, batch_size, summary_writer):
    """Learning is happening here."""

    data_holder = DataManager(summary_writer)

    env, observation = make_env(env_name)
    prev_reward = None
    train_step_count = 0
    while True:

        if train_step_count == 200:
            train_step_count = 0
            observation = env.reset()

            data_holder.next_episode()

            if data_holder.record_counter >= batch_size:

                with summary_writer.as_default():
                    policy.train_step(
                        data_holder.observations(),
                        np.vstack(data_holder.labels()),
                        np.vstack(data_holder.rewards_discounted()),
                        step=tf.constant(data_holder.episode_number, dtype=tf.int64)
                    )

                data_holder.next_batch()
            elif data_holder.record_counter >= batch_size:
                data_holder.next_batch()

            time.sleep(0.01)

        env.render()

        policy_input = observation
        # print("observation: ", policy_input)
        #########################
        #epislon greedy sampling#
        #########################
        epi = 0.2
        if np.random.rand() < epi:
            action = tf.random.uniform([4,], minval=-1., maxval=1., dtype=np.float32)
        else:
            action = policy.sample_action(policy_input)
        # step the environment and get new measurements
        # print("action: ", action)
        observation, reward, done, _ = env.step(action)
        train_step_count = train_step_count + 1
        data_holder.record_data(policy_input, reward, action)

        if prev_reward != reward:
            data_holder.log_summary()
            prev_reward = reward






if __name__ == "__main__":

    tf.random.set_seed(101)
    summary_writer = tf.summary.create_file_writer("./current.log")


    policy = Policy(4)
    # policy.load("./current.w")
    try:
        learning("pick-place-v2", policy, 32, summary_writer)
    except (KeyboardInterrupt):
        policy.save("./current.w")



# def discount_rewards(reward_his, gamma=.99):

#     discounted_r = np.zeros_like(reward_his)
#     running_add = 0

#     for i in reversed(range(0, reward_his.size)):
#         if reward_his[i] != 0:
#             running_add = 0
#         running_add = running_add * gamma + reward_his[i]
#         discounted_r[i] = running_add


#     return discounted_r



# def ortho_init(scale=1.0):
#     """Orthogonal weight initialization"""

#     def _ortho_init(shape, dtype, partition_info=None):  # pylint: disable=unused-argument
#         # lasagne ortho init for tf
#         shape = tuple(shape)
#         if len(shape) == 2:
#             flat_shape = shape
#         elif len(shape) == 4:  # assumes NHWC
#             flat_shape = (np.prod(shape[:-1]), shape[-1])
#         else:
#             raise NotImplementedError
#         a_val = np.random.normal(0.0, 1.0, flat_shape)
#         u_val, _, v_val = np.linalg.svd(a_val, full_matrices=False)
#         q_val = u_val if u_val.shape == flat_shape else v_val  # pick the one with the correct shape
#         q_val = q_val.reshape(shape)
#         return (scale * q_val[:shape[0], :shape[1]]).astype(np.float32)

#     return _ortho_init

# def surrogate_loss(logits_softmax, action_hist, reward_hist):
#     """Surrogate loss"""

#     loss = - tf.reduce_sum(
#         tf.math.multiply(
#             tf.math.multiply(reward_hist, action_hist), tf.math.log(logits_softmax + 0.0001)))

#     return loss


# class NatureCNN(Model):
#     """Nature CNN Policy network"""

#     def __init__(self, nb_outputs):
#         super(NatureCNN, self).__init__()

#         # self.flatten = Flatten()

#         self.dense_1 = Dense(
#             units=512, kernel_initializer=ortho_init(np.sqrt(2)), activation='relu')

#         self.dense_2 = Dense(
#             units=512, kernel_initializer=ortho_init(np.sqrt(2)), activation='relu')

#         self.logits = Dense(
#             units=nb_outputs, kernel_initializer=ortho_init(np.sqrt(2)), activation='tanh')

#         # self.logits_softmax = Softmax()

#     def call(self, inputs):  # pylint: disable=arguments-differ

#         ret = tf.cast(inputs, tf.float32)
#         ret = self.dense_1(ret)
#         ret = self.dense_2(ret)
#         ret = self.logits(ret)

#         return ret


# class Policy(object):
#     """Three Dense layers"""

#     def __init__(self, n_act, learing_rate=1e-3, decay=0.99):
#         """
#         ARGS:
#             learing_rate: learning rate
#             decay: RMSprop decay
#         """
#         """
#         dependent on system
#         """
#         self.model = NatureCNN(n_act)

#         self.optimizer = tf.keras.optimizers.RMSprop(
#             learning_rate=learing_rate,
#             rho=decay
#         )

#     def train_step(self, observations, actions, advantages, step=None):
#         """Make a single training step."""
#         with tf.GradientTape() as tape:

#             # training=True is only needed if there are layers with different
#             # behavior during training versus inference (e.g. Dropout).
#             predictions = self.model.call(observations)
#             loss = surrogate_loss(predictions, actions, advantages)

#         if step is not None:
#             tf.summary.scalar('loss', loss, step=step)

#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

#         return predictions 
 

#     def sample_action(self, observation):
#         """Sample action from observation.
#         Return index of action.
#         """

#         observations = tf.expand_dims(observation,axis = 0)
#         # observations = tf.reshape(observation,(4,observation.shape[0]))
#         action = self.model.call(observations)

#         action = tf.squeeze(action, axis=0)
#         print("action: ", action)
#         # LOGGER.info("action: %s", action)
#         # action = tf.argmax(input=action, axis=0)
#         return action

#     def save(self, checkpoint_path):
#         """Save Tesnroflow checkpoint."""

#         self.model.save_weights(checkpoint_path)

#         LOGGER.info('Checkpoint saved %s', checkpoint_path)

#     def load(self, checkpoint_path):
#         """Load Tesnsorflow checkpoint."""

#         self.model.load_weights(checkpoint_path)

#         LOGGER.info('Weights loaded from checkpoint file: %s', checkpoint_path)
