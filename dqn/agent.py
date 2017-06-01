import numpy as np
from .replay_memory import ReplayMemory
from .q import Q
from tqdm import tqdm


class Agent(object):
    def __init__(self, config, environment):
        self.env = environment
        self.config = config
        self.epsilon = config.epsilon
        self.copy_estimators = config.C
        self.replay_memory = ReplayMemory(self.config, environment)  # initialize replay memory
        self.Q = self.Q_hat = Q(config, environment, self.replay_memory)  # initialize Q functions

    def train(self):
        for episode in tqdm(range(self.config.episode_size)):
            self.env.reset_environment()  # reset environment each episode
            self.replay_memory.reset()  # reset replay memory
            for timestep in range(len(self.env.get_date_range())):
                s = self.env.get_current_state()
                action = self.__choose_action__(s)
                s_prime, reward, terminal = self.env.act(action)
                self.replay_memory.store_transition(s, action, reward, s_prime)
                minibatch_samples = self.replay_memory.sample_replays()
                y = self.Q.compute_targets(minibatch_samples)
                # y = self.Q_hat.compute_targets(minibatch_samples)
                self.Q.train_network(y, minibatch_samples)
                # if timestep % self.copy_estimators == 0:  # every timestep C, reset Q hat to Q
                #     self.Q_hat.set_weights(self.Q.get_weights())

    def test(self):
        cum_reward = 0
        self.env.reset_environment()  # reset environment
        for timestep in tqdm(range(len(self.env.get_date_range()))):
            s = self.env.get_current_state()
            action = self.__choose_action__(s)
            s_prime, reward, terminal = self.env.act(action)
            cum_reward += reward
        print(cum_reward)

    def __choose_action__(self, s):
        if self.__with_probability__(self.epsilon):  # with probability epsilon, return random action
            action = np.random.rand(self.env.num_stocks)
            action /= action.sum()
        else:
            action = self.Q.best_action(s)
        return action

    def __with_probability__(self, epsilon):
        random = np.random.random_sample()
        return True if random <= epsilon else False
