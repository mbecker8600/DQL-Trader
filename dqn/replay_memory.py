import numpy as np
import pandas as pd
from config import AgentConfig


class ReplayMemory:
    def __init__(self, config, env):
        self.env = env
        self.n_history = config.n_history
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.replay_memory = pd.DataFrame(np.nan, index=range(self.memory_size), columns=['s', 'a', 'r', 's_prime'])
        self.counter = 0

        self.__buffer_memory__()  # buffer the memory  with an initial set of states in the history with dummy actions

    def __buffer_memory__(self):
        s = self.env.get_n_history_state()
        a = np.random.rand(self.n_history, self.env.get_num_stocks())
        a /= a.sum()
        self.replay_memory['s'] = self.replay_memory['s'].astype(object)  # set to allow 2d (num_history, num_input)
        self.replay_memory['a'] = self.replay_memory['a'].astype(object)  # set to allow 2d (num_history, num_stocks)
        self.replay_memory.ix[self.counter] = (s, a, None, None)
        self.counter += 1

    def store_transition(self, s, a, r, s_prime):
        prev_transition = self.get_previous_state_action()
        prev_state = pd.DataFrame(prev_transition['s']).shift(1)
        prev_action = pd.DataFrame(prev_transition['a']).shift(1)
        prev_state.ix[0] = s
        prev_action.ix[0] = a
        self.replay_memory.ix[self.counter % self.memory_size] = (s, a, r, s_prime)
        self.counter += 1

    def get_previous_state_action(self):
        return self.replay_memory.ix[(self.counter - 1) % self.memory_size][['s', 'a']]

    def sample_replays(self):
        replay_memory = self.replay_memory.dropna()
        rand_idx = np.random.randint(replay_memory.shape[0], size=self.batch_size)
        return self.replay_memory.ix[rand_idx]

    def reset(self):
        self.replay_memory[:] = np.nan
        self.counter = 0
        self.__buffer_memory__()  # buffer the memory  with an initial set of states in the history with dummy actions

# testing purposes
if __name__ == '__main__':
    from dqn.environment import Environment
    from datetime import datetime

    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config)
    replay_memory = ReplayMemory(config, env)
    replay_memory.store_transition(np.random.rand(1, 12575), np.random.rand(1, 503), 1, 1)
    print(replay_memory.sample_replays())
