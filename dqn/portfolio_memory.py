import numpy as np


class PortfolioMemory:

    def __init__(self, n_instruments, n_history):
        self.portfolio_allocations = np.zeros((n_history, n_instruments))

    def add_memory(self, allocations):
        tmp = np.roll(self.portfolio_allocations, 1, 0)
        tmp[0, :] = np.array(allocations)
        self.portfolio_allocations = tmp

# testing purposes
if __name__ == '__main__':
    portfolio_memory = PortfolioMemory(4, 5)
    portfolio_memory.add_memory([.1, .1, .5, .3])
    portfolio_memory.add_memory([.5, .1, .1, .3])
    portfolio_memory.add_memory([.1, .5, .1, .3])
