import numpy as np
import pandas as pd
from config import AgentConfig


class ReplayMemory:
    def __init__(self, config, initial_state, n_syms):
        self.n_history = config.n_history
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.replay_memory = pd.DataFrame(np.nan, index=range(self.memory_size), columns=['s', 'a', 'r', 's_prime', 'terminal'])
        self.counter = 0

        self.__buffer_memory__(initial_state, n_syms)  # buffer the memory  with an initial set of states in the history with dummy actions

    def __buffer_memory__(self, initial_state, n_syms):
        a = np.random.rand(self.n_history, n_syms * 2)
        a /= a.sum()
        self.replay_memory['s'] = self.replay_memory['s'].astype(object)  # set to allow 2d (num_history, num_input)
        self.replay_memory['a'] = self.replay_memory['a'].astype(object)  # set to allow 2d (num_history, num_stocks)
        self.replay_memory['s_prime'] = self.replay_memory['s_prime'].astype(object)  # set to allow 2d (num_history, num_stocks)
        self.replay_memory.ix[self.counter] = (initial_state, a, np.nan, np.nan, False)
        self.counter += 1

    def store_transition(self, s, a, r, s_prime):
        prev_transition = self.get_previous_state_action()
        prev_state = pd.DataFrame(prev_transition['s']).shift(1)
        prev_action = pd.DataFrame(prev_transition['a']).shift(1)
        prev_state.ix[0] = s
        prev_action.ix[0] = a
        prev_s_prime = prev_state.shift(1)
        prev_s_prime.ix[0] = s_prime
        terminal = True if prev_s_prime.isnull().values.any() else False
        self.replay_memory.ix[self.counter % self.memory_size] = (prev_state.values, prev_action.values, r, prev_s_prime.values, terminal)
        self.counter += 1

    def get_previous_state_action(self):
        return self.replay_memory.ix[(self.counter - 1) % self.memory_size][['s', 'a']]

    def sample_replays(self):
        replay_memory = self.replay_memory.dropna(how='any')
        rand_idx = np.random.choice(replay_memory.index.tolist(), size=self.batch_size)
        return replay_memory.ix[rand_idx]

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
