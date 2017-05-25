import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from datetime import datetime
from config import AgentConfig


class Q:
    def __init__(self, config, env):
        self.env = env

        # network params
        self.n_outputs = 1
        self.n_inputs = env.get_num_stocks() * env.get_num_indicators()
        self.n_hidden = config.n_hidden
        self.n_history = config.n_history

        # Define weights/biases
        self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_outputs]))
        self.biases = tf.Variable(tf.random_normal([self.n_outputs]))

        # Start tf session
        self.sess = tf.Session()

        # Define hyperparameters
        self.learning_rate = config.learning_rate

    def best_action(self, s):
        x = tf.placeholder("float", [None, self.n_history, self.n_inputs])
        pred = self.__RNN__(x, self.weights, self.biases)
        init = tf.global_variables_initializer()  # Initializing the variables
        self.sess.run(init)
        batch_x = np.random.rand(1, self.n_history, self.n_inputs)
        reward = self.sess.run(pred, feed_dict={x: batch_x})
        return reward

    def __optimize_reward__(self, a, s):
        pass

    def max_reward(self, s):
        pass

    def train_network(self, y, samples):
        pass

    def compute_targets(self, samples):
        pass

    def __RNN__(self, X, weights, biases):
        X = tf.unstack(X, self.n_history, 1)
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    def set_weights(self, new_weights):
        self.weights = new_weights

    def get_weights(self):
        return self.weights

