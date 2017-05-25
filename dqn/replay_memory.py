import numpy as np
import pandas as pd
from config import AgentConfig

class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.replay_memory = pd.DataFrame(np.nan, index=range(self.memory_size), columns=['s', 'a', 'r', 's_prime'])
        self.counter = 0

    def store_transition(self, s, a, r, s_prime):
        self.replay_memory.ix[self.counter % self.memory_size] = (s, a, r, s_prime)
        self.counter += 1

    def sample_replays(self):
        replay_memory = self.replay_memory.dropna()
        rand_idx = np.random.randint(replay_memory.shape[0], size=self.batch_size)
        return self.replay_memory.ix[rand_idx]

    def reset(self):
        self.replay_memory[:] = np.nan

# testing purposes
if __name__ == '__main__':
    config = AgentConfig()
    replay_memory = ReplayMemory(config)
    replay_memory.store_transition(1, 1, 1, 1)
    replay_memory.store_transition(2, 2, 2, 2)
    replay_memory.store_transition(3, 3, 3, 3)
    replay_memory.store_transition(4, 4, 4, 4)
    print(replay_memory.sample_replays())
