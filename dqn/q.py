import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from scipy.optimize import minimize
from dqn.model import Model


class Q:
    def __init__(self, config, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self.config = config

        # network params
        self.n_outputs = 1
        self.n_inputs = env.get_num_stocks() * env.get_num_indicators() + env.get_num_stocks()
        self.n_hidden = config.n_hidden
        self.n_history = config.n_history

        # Define placeholders and model
        self.x_placeholder = tf.placeholder("float", [None, self.n_history, self.n_inputs])
        self.y_placeholder = tf.placeholder("float", [None, 5, 5])
        self.model = Model(self.x_placeholder, self.y_placeholder, config, env)

        # Start tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()  # Initializing the variables
        self.sess.run(init)

        # Define hyperparameters
        self.learning_rate = config.learning_rate

    def best_action(self, s):
        previous_transitions = self.replay_memory.get_previous_state_action()
        previous_action = previous_transitions[1][0]

        # find the action that maximizes the reward
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1.0) for _ in range(len(previous_action))]
        res = minimize(self.__optimize_reward__, x0=previous_action, args=(s,), bounds=bounds, constraints=cons, method='SLSQP', options={'disp': True})
        return res.x

    def max_reward(self, s):
        pass

    def train_network(self, y, samples):
        pass

    def compute_targets(self, samples):
        pass

    def set_weights(self, new_weights):
        self.weights = new_weights

    def get_weights(self):
        return self.weights

    def __build_input__(self, s, a, previous_transition):
        previous_states = previous_transition[0]  # (n_history, n_stocks * n_indicators)
        previous_actions = previous_transition[1]  # (n_history, n_stocks)

        # roll off last transition to make room for the new state and action
        states = np.roll(previous_states, 1)
        actions = np.roll(previous_actions, 1)

        # add new action and state to the input
        states[0] = s.ravel()
        actions[0] = a.ravel()

        # return concatenated input
        return np.concatenate((states, actions), axis=1)

    def __optimize_reward__(self, a, s):
        input = self.__build_input__(s, a, self.replay_memory.get_previous_state_action())
        batch_x = np.reshape(input, (1, self.n_history, self.n_inputs))
        return -self.sess.run(self.model.prediction, {self.x_placeholder: batch_x})[0][0]


