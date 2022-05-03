
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax
from tensorflow.keras import Model
import numpy as np
import random
import gym

import tflearn

import logging
import datetime
from os.path import dirname, join, abspath
import time
import metaworld

def get_logger(logger_name):
    """Get logger with predefined settings."""

    logging.basicConfig(level=logging.DEBUG, format='[%(name)s] %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(logger_name)

    return logger


LOGGER = get_logger('PG')

## ref: https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py

from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# class DataManager:
#     """Record and hold data required during training."""

#     def __init__(self, summary_writer):

#         self.summary_writer = summary_writer

#         self._labels = []
#         self._rewards = []
#         self._observations = []

#         self.record_timestamp = None

#         # Load the step from previous trainings
#         self.episode_number = tf.summary.experimental.get_step()
#         if not self.episode_number:
#             self.episode_number = 0
#             self.episode_number_batch = 0

#         self.start_time = time.time()
#         self.record_counter = 0
#         self._last_record_count = 0

#     def record_data(self, observation, reward, action):
#         """Record data for historical purposes."""

#         if not self._rewards:
#             self.record_timestamp = time.time()

#         self._observations.append(observation)
#         self._rewards.append(reward)


#         label = action


#         self._labels.append(label)

#         self.record_counter += 1

#     # @property
#     def rewards_discounted(self):
#         """Compute the discounted reward backwards through time."""

#         reward_his = discount_rewards(self.rewards())
#         # standardize the rewards to be unit normal
#         # (helps control the gradient estimator variance)
#         reward_his -= np.mean(reward_his)
#         tmp = np.std(reward_his)
#         if tmp > 0:
#             reward_his /= tmp  # fix zero-divide

#         return reward_his

#     # @property
#     def rewards(self):
#         """Return reward history in numpy array."""
#         return np.array(self._rewards, dtype=np.float32)

#     # @property
#     def rewards_episode(self):
#         """Return reward history in numpy array."""
#         return np.array(
#             self._rewards[self._last_record_count: self.record_counter], dtype=np.float32)

#     # @property
#     def labels(self):
#         """Return reward history in numpy array."""
#         return np.array(self._labels, dtype=np.float32)

#     # @property
#     def observations(self):
#         """Return reward history in numpy array."""
#         return np.array(self._observations, dtype=np.uint8)

#     # @property
#     def record_counter_episode(self):
#         """Return record counter for single episode."""
#         return self.record_counter - self._last_record_count

#     def log_summary(self):
#         """Print out in logs summary about training."""

#         current_time = time.time()

#         fps = 0
#         if self.record_counter > 2:  # if observation is not empty
#             fps = self.record_counter / (current_time - self.record_timestamp)

#         LOGGER.debug("%s. T[%.2fs] FPS: %.2f, Reward Sum: %s",
#                      self.episode_number, current_time - self.start_time, fps,
#                      sum(self._rewards[self._last_record_count: self.record_counter]))

#     def next_batch(self):
#         """Clear gathered data and prepare to for next episode."""
#         batch_reward = sum(self._rewards) / self.episode_number_batch
#         with self.summary_writer.as_default():
#             tf.summary.scalar('batch_reward', batch_reward, step=self.episode_number)

#         self._labels = []
#         self._rewards = []
#         self._observations = []
#         self.record_counter = 0
#         self._last_record_count = 0
#         self.episode_number_batch = 0
#         self.record_timestamp = None

#     def next_episode(self):
#         """Next Episode."""

#         episode_rewards = sum(self._rewards[self._last_record_count: self.record_counter])
#         episode_record_counter = self.record_counter - self._last_record_count

#         with self.summary_writer.as_default():
#             tf.summary.scalar('reward', episode_rewards, step=self.episode_number)
#             tf.summary.scalar('number of records', episode_record_counter, step=self.episode_number)

#         self.episode_number += 1
#         self.episode_number_batch += 1
#         self._last_record_count = self.record_counter

#         tf.summary.experimental.set_step(self.episode_number)


def discount_rewards(reward_his, gamma=.99):

    discounted_r = np.zeros_like(reward_his)
    running_add = 0

    for i in reversed(range(0, reward_his.size)):
        if reward_his[i] != 0:
            running_add = 0
        running_add = running_add * gamma + reward_his[i]
        discounted_r[i] = running_add


    return discounted_r


# def set_global_seeds(seed):
#     """Set global seeds for random generators."""

#     tf.random.set_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)



###### CNN Model 

# def conv(n_f, r_f, stride, activation, pad='valid', init_scale=1.0, data_format='channels_last'):
#     """Conv2D wrapper."""

#     layer = Conv2D(
#         filters=n_f, kernel_size=r_f, strides=stride, padding=pad, activation=activation,
#         data_format=data_format, kernel_initializer=ortho_init(init_scale)
#     )

#     return layer


class ActorNetwork(object, env):

    self.s_dim = env.observation_space.shape[0]
    self.a_dim = env.action_space.shape[0]
    self.learning_rate = learning_rate
    self.tau = tau
    self.batch_size = batch_size



    self.dense_1 = Dense(
        units=512, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='relu')
    self.dense_2 = Dense(
        units=512, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='relu')
    self.logits = Dense(
        units=self.a_dim, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='tanh')

    self.network_params = tf.trainable_variables()

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
