import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import random
import tensorflow_probability as tfp

from policies.gaussian_policy import Actor
from wrappers.critic_wrapper import Critic_Wrapper
from replay_buffer.r_buffer import Buffer

class PPO:
    
    def __init__(self, env, observation_dimensions, action_dimensions, action_bound, horizon,
                 minibatch_size=256, gamma=0.99, lam=0.95, diagnostic_length=1, lr=3e-4):
        
        self.env = env
        self.actor = Actor(action_dimensions, action_bound)
        self.critic_gen = Critic_Wrapper(observation_dimensions)
        self.critic = self.critic_gen.get_critic()
        self.buffer = Buffer(observation_dimensions, action_dimensions, horizon, minibatch_size, gamma, lam)
        
        self.p_opt= tf.keras.optimizers.Adam(learning_rate=lr,
                                                            )
        self.v_opt= tf.keras.optimizers.Adam(learning_rate=lr,
                                                            )
        self.clip_epsilon = 0.2
        
        self.diagnostics_buffer = []
        self.diagno_index = 0
        self.diagno_length = diagnostic_length
        
        self.gamma, self.lam, self.horizon = gamma, lam, horizon
        
    def train(self, iterations, epochs=20):
        
        for i in range(iterations):
            
            obs = self.env.reset()
            
            for t in range(self.horizon):
                
                tf_obs = tf.expand_dims(obs, 0)
                a, log_a, h_a = self.actor(tf_obs)
                a=a[0]
            
                obs_new, r, d, _ = self.env.step(a)
                
                self.buffer.store(obs, a, r, log_a, d)
                
                if d:
                    obs = self.env.reset()
                else:
                    obs = obs_new

            for _ in range(epochs):
                (
                    obs_b,
                    a_b,
                    adv_b,
                    ret_b,
                    log_b,
                    r_b,
                ) = self.buffer.batch_sample(self.critic)
                p_l, c_l = self.update(obs_b, adv_b, log_b, ret_b)
                self.record_diagnostics(["policy loss: ", np.array(p_l), "value loss: ", np.array(c_l)])
                
            self.show_diagnostics()    
            self.buffer.clear()
    
    @tf.function
    def update(self, obs_b, adv_b, log_b, ret_b):
        with tf.GradientTape() as tape:
            a, log_a, h_a = self.actor(obs_b)
            ratio = tf.exp(log_a - log_b)
            c_ratio = tf.clip_by_value(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon)

            rt_at = tf.minimum(tf.math.multiply(ratio, tf.cast(adv_b, tf.float32)), 
                               tf.math.multiply(c_ratio, tf.cast(adv_b, tf.float32)))
            
            L_theta_clip = -tf.reduce_mean(rt_at+h_a)
        J_theta_clip = tape.gradient(L_theta_clip, self.actor.trainable_variables)
        self.p_opt.apply_gradients(zip(J_theta_clip, self.actor.trainable_variables))
        
        with tf.GradientTape() as tape1:
            v_theta = tf.squeeze(self.critic(obs_b))
            v_mse = tf.reduce_mean((v_theta - ret_b)**2)
        J_phi = tape1.gradient(v_mse, self.critic.trainable_variables)
        self.v_opt.apply_gradients(zip(J_phi, self.critic.trainable_variables))
        
        return L_theta_clip, v_mse
        
    def record_diagnostics(self, data):
        if len(self.diagnostics_buffer) == self.diagno_length:
            self.diagnostics_buffer[self.diagno_index] = data

        if len(self.diagnostics_buffer) < self.diagno_length:
            self.diagnostics_buffer.append(data)
        self.diagno_index = (self.diagno_index+1)%self.diagno_length
    def show_diagnostics(self):
        for i in range(len(self.diagnostics_buffer)):
            print(self.diagnostics_buffer[(self.diagno_index+i)%len(self.diagnostics_buffer)], flush=True)
            
    def save_weights(self, a_path, c_path):
        self.actor.save_weights(a_path)
        print("Saved actor weights", flush=True)
        self.critic.save_weights(c_path)
        print("Saved critic weights", flush=True)

    def load_weights(self, a_path, c_path):
        try:
            self.actor.load_weights(a_path)
            print("Loaded actor weights", flush=True)
            self.critic.load_weights(c_path)
            print("Loaded critic weights", flush=True)
        except ValueError:
            print("ERROR: Please make sure weights are saved as .ckpt", flush=True)
            
        
    def eval_rollout(self, eval_env):
        eps_r = 0
        eval_obs = eval_env.reset()
        
        while True:
#             eval_env.render()

            tf_eval_obs = tf.expand_dims(tf.convert_to_tensor(eval_obs), 0)

            eval_a, eval_log_a, eval_h_a = self.actor(tf_eval_obs, eval_mode=True)

            eval_a = eval_a[0]

            eval_obs_new, eval_r, eval_d, _ = eval_env.step(eval_a)

            eps_r += eval_r

            if eval_d:
                break
                
            eval_obs = eval_obs_new
            
        print("rollout episodic reward: ", eps_r, flush=True)