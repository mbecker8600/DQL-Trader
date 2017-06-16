import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from dqn.actor_model import ActorModel
from dqn.critic_model import CriticModel


class Q:
    def __init__(self, config, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self.config = config

        # global parameters
        self.n_history = config.n_history
        self.n_stocks = env.get_num_stocks()
        self.n_actions = env.get_num_stocks()
        self.n_features = env.get_num_stocks() * env.get_num_indicators()

        # network params
        self.n_critic_outputs = 1
        self.n_actor_outputs = env.get_num_stocks()
        self.n_hidden = config.n_hidden

        # Define actor placeholders
        self.x_actor_placeholder = tf.placeholder("float", [None, self.n_history, self.n_features])
        self.action_gradient_placeholder = tf.placeholder("float", [None, self.n_actions])

        # Define critic placeholders
        self.s_critic_placeholder = tf.placeholder("float", [None, self.n_history, self.n_features])
        self.a_critic_placeholder = tf.placeholder("float", [None, self.n_history, self.n_actions])
        self.y_critic_placeholder = tf.placeholder("float", [None, self.n_critic_outputs])

        # Define models
        self.actor_model = ActorModel(self.x_actor_placeholder, self.action_gradient_placeholder, config, env)
        self.critic_model = CriticModel(self.s_critic_placeholder, self.a_critic_placeholder, self.y_critic_placeholder, config, env)

        # Start tf session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if not config.resume_training:
            init = tf.global_variables_initializer()  # Initializing the variables
            self.sess.run(init)
        self.saver = tf.train.Saver()
        self.checkpoint_loc = config.checkpoint_loc
        if config.resume_training:
            self.saver.restore(self.sess, "C:\\tmp\\dqn\\model.ckpt")

        # Define hyperparameters
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        self.minmax_scaler = preprocessing.MinMaxScaler()

    def best_action_batch(self, s):
        assert len(s.shape) is 3, \
            "The state should always be in the form (batch_size, n_history, n_features). " \
            "Input shape is {}".format(len(s.shape))
        assert s.shape[1] == self.n_history, \
            "The history should be equal to {}, but the input was {}. ".format(self.n_history, s.shape[1])
        assert s.shape[2] == self.n_features, \
            "The number of features should be equal to {}, but the input was {}. ".format(self.n_features, s.shape[2])

        action = self.__predict_action__(s)
        action = self.minmax_scaler.fit_transform(action.T)
        action /= action.T.sum()
        return action.T

    def best_action(self, s):
        assert len(s.shape) is 2, \
            "The state should always be in the form (n_history, n_features). " \
            "Input shape is {}".format(len(s.shape))

        if s.shape[0] == 1:  # if the state only holds the current day's history, grab the previous history
            input = self.__build_actor_input(s, self.replay_memory.get_previous_state_action())
        s = np.reshape(input, (1, self.n_history, self.n_features))  # reshape to tensorflow format

        return self.__predict_action__(s)[0]

    def get_action_gradients(self, samples):
        batch_s = np.array([samples['s'].values[i] for i in range(len(samples['s'].values))])
        batch_a = np.array([samples['a'].values[i] for i in range(len(samples['a'].values))])
        return self.sess.run(self.critic_model.action_gradients, {self.s_critic_placeholder: batch_s, self.a_critic_placeholder: batch_a})[0][:, 0]

    def train_actor_network(self, gradients, samples):
        batch_s = np.array([samples['s'].values[i] for i in range(len(samples['s'].values))])
        self.sess.run(self.actor_model.optimize,
                      {self.x_actor_placeholder: batch_s, self.action_gradient_placeholder: gradients})
        pass

    def train_critic_network(self, y, samples):
        batch_s = np.array([samples['s'].values[i] for i in range(len(samples['s'].values))])
        batch_a = np.array([samples['a'].values[i] for i in range(len(samples['a'].values))])
        self.sess.run(self.critic_model.optimize, {self.s_critic_placeholder: batch_s, self.a_critic_placeholder: batch_a, self.y_critic_placeholder: y})
        self.saver.save(self.sess, "C:\\tmp\\dqn\\model.ckpt")

    def compute_targets(self, samples):
        batch_s_prime = np.array([samples['s_prime'].values[i] for i in range(len(samples['s_prime'].values))])
        new_actions = self.best_action_batch(batch_s_prime)
        batch_actions = np.array([samples['a'].values[i] for i in range(len(samples['a'].values))])
        batch_actions = np.roll(batch_actions, 1, axis=1)  # roll off the old actions
        batch_actions[:, 0] = new_actions  # put the new actions in it's spot
        return self.__predict_reward__(batch_s_prime, batch_actions)

    def set_weights(self, new_weights):
        self.weights = new_weights

    def get_weights(self):
        return self.weights

    def __build_actor_input(self, s, previous_transition):
        previous_states = previous_transition[0]  # (n_history, n_stocks * n_indicators)

        # roll off last transition to make room for the new state and action
        states = np.roll(previous_states, 1)

        # add new action and state to the input
        states[0] = s.ravel()

        return states

    def __predict_reward__(self, s, a):
        return self.sess.run(self.critic_model.prediction,
                             {self.s_critic_placeholder: s, self.a_critic_placeholder: a})

    def __predict_action__(self, s):
        return self.sess.run(self.actor_model.prediction, {self.x_actor_placeholder: s})


