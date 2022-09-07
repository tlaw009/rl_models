import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from tensorflow.keras import Model
# import matplotlib.pyplot as plt
import random
import tensorflow_probability as tfp
import glfw

from wrappers.critic_wrapper import Critic_Wrapper
from policies.gaussian_policy import Actor
from replay_buffer.r_buffer_sac import Buffer
from wrappers.robosuite_wrapper import Robosuite_Wrapper

class SAC:
    
    def __init__(self, env, observation_dimensions, action_dimensions, action_bound, buffer_capacity,
                 minibatch_size=256, gamma=0.99, tau=0.95, lr=3e-4):
        
        self.env = env
        tf.debugging.enable_check_numerics()
        self.a = Actor(action_dimensions, action_bound)
        self.c_gen = Critic_Wrapper(observation_dimensions, action_dimensions)
        self.c1 = self.c_gen.get_critic()
        self.c2 = self.c_gen.get_critic()
        self.tc1 = self.c_gen.get_critic()
        self.tc2 = self.c_gen.get_critic()
        
        self.tc1.set_weights(self.c1.get_weights())
        self.tc2.set_weights(self.c2.get_weights())

        self.te = -np.prod(action_dimensions)
        self.alpha = tf.Variable(0.0, dtype=tf.float32)
        
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.c1_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.c2_opt = tf.keras.optimizers.Adam(learning_rate=lr)                                                  
        self.alpha_opt = tf.keras.optimizers.Adam(learning_rate=lr)   
        
        self.buffer = Buffer(observation_dimensions, action_dimensions, buffer_capacity, minibatch_size)
        
        self.gamma, self.tau = gamma, tau
        
    def train(self, max_env_step):
        t = 0
        a_losses = []
        c1_losses = []
        c2_losses = []
        alpha_losses = []
        while t < max_env_step:
            p_s = self.env.reset()

            while True:
                a, log_a = self.a(tf.expand_dims(p_s, 0))
                a=a[0]
                s, r, d, _ = self.env.step(a)
                end = 0 if d else 1
                
                self.buffer.record((p_s, a, r, s, end))
                data = self.buffer.sample()
                
                a_loss, c1_loss, c2_loss, alpha_loss = self.update(data)
                
                a_losses.append(a_loss.numpy())
                c1_losses.append(c1_loss.numpy())
                c2_losses.append(c2_loss.numpy())
                alpha_losses.append(alpha_loss.numpy())
                
                t = t+1
                
                if d:
                    break
                p_s = s
                
        print("Per {:04d} Steps".format(max_env_step), "Policy Avg. Loss: ", np.mean(a_losses), 
              ", Critic 1 Avg. Loss: ",  np.mean(c1_losses), 
              ", Critic 2 Avg. Loss: ",  np.mean(c2_losses), 
              ", Alpha 1 Avg. Loss: ",  np.mean(alpha_losses), flush=True)


    @tf.function
    def update(self, data):
        s_b, a_b, r_b, ns_b, d_b = data
        with tf.GradientTape() as tape_c1, tf.GradientTape() as tape_c2:
            q1 = self.c1([s_b, a_b])
            q2 = self.c2([s_b, a_b])
            na, nlog_a = self.a(ns_b, training=True)
            
            tq1 = self.tc1([ns_b, na])
            tq2 = self.tc2([ns_b, na])
            
            min_qt = tf.math.minimum(tq1,tq2)
            
            soft_qt = min_qt - (self.alpha*nlog_a)
            
            y = tf.stop_gradient(r_b+self.gamma*d_b*tf.cast(soft_qt, dtype=tf.float64))
            
            L_c1 = 0.5*tf.reduce_mean((y-tf.cast(q1, dtype=tf.float64))**2)
            L_c2 = 0.5*tf.reduce_mean((y-tf.cast(q2, dtype=tf.float64))**2)
        c1_grad = tape_c1.gradient(L_c1, self.c1.trainable_variables)
        c2_grad = tape_c2.gradient(L_c2, self.c2.trainable_variables)
        
        self.c1_opt.apply_gradients(zip(c1_grad, self.c1.trainable_variables))
        self.c2_opt.apply_gradients(zip(c2_grad, self.c2.trainable_variables))
        
        for (tc1w, c1w) in zip(self.tc1.variables, self.c1.variables):
            tc1w.assign(tc1w*self.tau + c1w*(1.0-self.tau))
        for (tc2w, c2w) in zip(self.tc2.variables, self.c2.variables):
            tc2w.assign(tc2w*self.tau + c2w*(1.0-self.tau))
            
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_alpha:
            a, log_a = self.a(s_b, training=True)
            qa1 = self.c1([s_b, a])
            qa2 = self.c2([s_b, a])
            
            soft_qa = tf.math.minimum(qa1,qa2)

            L_a = -tf.reduce_mean(soft_qa-self.alpha*log_a)
            L_alpha = -tf.reduce_mean(self.alpha*tf.stop_gradient(log_a + self.te))
        grad_a = tape_a.gradient(L_a, self.a.trainable_variables)
        grad_alpha = tape_alpha.gradient(L_alpha, [self.alpha])
        self.a_opt.apply_gradients(zip(grad_a, self.a.trainable_variables))
        self.alpha_opt.apply_gradients(zip(grad_alpha, [self.alpha]))
        
        return L_a, L_c1, L_c2, L_alpha
    
    def save_weights(self, dir_path):
        cp = tf.train.Checkpoint(step=self.alpha)
        self.a.save_weights(dir_path+"/a.ckpt")
        print("Saved actor weights", flush=True)
        self.c1.save_weights(dir_path+"/c1.ckpt")
        print("Saved critic 1 weights", flush=True)
        self.c2.save_weights(dir_path+"/c2.ckpt")
        print("Saved critic 2 weights", flush=True)
        cp.save(dir_path+"/alpha")
        print("Saved alpha weights", flush=True)

    def load_weights(self, dir_path):
        try:
            cp = tf.train.Checkpoint(step=self.alpha)
            self.a.load_weights(dir_path+"/a.ckpt")
            print("Loaded actor weights", flush=True)
            self.c1.load_weights(dir_path+"/c1.ckpt")
            print("Loaded critic 1 weights", flush=True)
            self.c2.load_weights(dir_path+"/c2.ckpt")
            print("Loaded critic 2 weights", flush=True)
            cp.restore(dir_path+"/alpha-1")
            print("Loaded alpha weights", flush=True)
            self.tc1.set_weights(self.c1.get_weights())
            self.tc2.set_weights(self.c2.get_weights())

        except ValueError:
            print("ERROR: Please make sure weights are saved as .ckpt", flush=True)
            
    def eval_rollout(self, problem, rbs=False, render=False):
        eps_r = 0
        
        if rbs:
            domain, task, controller = problem
            eval_env = Robosuite_Wrapper(domain, task, controller, render)
        else:
            eval_env = gym.make(problem)
            
        eval_obs = eval_env.reset()

        while True:
            if render:
                eval_env.render()

            tf_eval_obs = tf.expand_dims(tf.convert_to_tensor(eval_obs), 0)

            eval_a, eval_log_a = self.a(tf_eval_obs, eval_mode=True)

            eval_a = eval_a[0]

            eval_obs_new, eval_r, eval_d, _ = eval_env.step(eval_a)

            eps_r += eval_r

            if eval_d:
                break
                
            eval_obs = eval_obs_new
        
        if render:
            if not rbs:
                glfw.destroy_window(eval_env.viewer.window)

        eval_env.close()
        print("rollout episodic reward: ", eps_r, flush=True)
        
        return eps_r
