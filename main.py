from datetime import datetime
from config import AgentConfig
from dqn.agent import Agent
import pytz
import warnings


if __name__ == '__main__':
    warnings.simplefilter("ignore", DeprecationWarning)

    config = AgentConfig()
    # env = Environment(sd, ed, config, datafile_loc='./fundretriever/snp500.h5')

    # parameters
    sd = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    ed = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    syms = ['GOOGL', 'AAPL', 'XOM', 'IBM']
    captial = 1000000

    agent = Agent(config, syms, captial)
    # agent.train(sd, ed)
    agent.test(sd, ed)

