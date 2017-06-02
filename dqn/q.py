import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from dqn.model import Model
from profilehooks import profile


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
        self.n_stocks = env.get_num_stocks()

        # Define placeholders and model
        self.x_placeholder = tf.placeholder("float", [None, self.n_history, self.n_inputs])
        self.y_placeholder = tf.placeholder("float", [None, self.n_outputs])
        self.model = Model(self.x_placeholder, self.y_placeholder, config, env)

        # Start tf session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if config.resume_training:
            init = tf.global_variables_initializer()  # Initializing the variables
            self.sess.run(init)
        self.saver = tf.train.Saver()
        self.checkpoint_loc = config.checkpoint_loc
        if not config.resume_training:
            self.saver.restore(self.sess, "C:\\tmp\\dqn\\model.ckpt")

        # Define hyperparameters
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        self.bounds = [(0, 1.0) for _ in range(self.n_stocks)]

    def best_action(self, s):
        previous_transitions = self.replay_memory.get_previous_state_action()
        previous_action = previous_transitions[1][0]

        # find the action that maximizes the reward
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = minimize(self.__optimize_reward__, x0=previous_action, args=(s,), bounds=self.bounds, constraints=cons, method='SLSQP')
        return res.x

    def train_network(self, y, samples):
        state_actions = samples[['s', 'a']].values
        batch_x = []
        for sample in range(state_actions.shape[0]):
            batch_x.append(np.concatenate((state_actions[sample][0], state_actions[sample][1]), axis=1))
        batch_x = np.array(batch_x)
        batch_y = np.reshape(y, (self.batch_size, self.n_outputs))
        self.sess.run(self.model.optimize, {self.x_placeholder: batch_x, self.y_placeholder: batch_y})
        self.saver.save(self.sess, "C:\\tmp\\dqn\\model.ckpt")

    def compute_targets(self, samples):
        y = []
        for index, sample in samples.iterrows():
            s_prime = sample['s_prime'][0]
            a_prime = self.best_action(s_prime)
            reward = sample['r'] + self.gamma * self.__predict_reward__(s_prime, a_prime)
            y.append(reward)
        return y


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

    def __predict_reward__(self, s, a):
        input = self.__build_input__(s, a, self.replay_memory.get_previous_state_action())
        batch_x = np.reshape(input, (1, self.n_history, self.n_inputs))
        return self.sess.run(self.model.prediction, {self.x_placeholder: batch_x})[0][0]

    def __optimize_reward__(self, a, s):
        input = self.__build_input__(s, a, self.replay_memory.get_previous_state_action())
        batch_x = np.reshape(input, (1, self.n_history, self.n_inputs))
        return -self.sess.run(self.model.prediction, {self.x_placeholder: batch_x})[0][0]


