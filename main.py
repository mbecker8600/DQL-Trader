from datetime import datetime
from config import AgentConfig
from dqn.agent import Agent
import pytz
import warnings
import pandas as pd


if __name__ == '__main__':
    warnings.simplefilter("ignore", DeprecationWarning)

    config = AgentConfig()
    # env = Environment(sd, ed, config, datafile_loc='./fundretriever/snp500.h5')

    # parameters
    sd = datetime(2014, 10, 1, 0, 0, 0, 0, pytz.utc)
    ed = datetime(2017, 7, 1, 0, 0, 0, 0, pytz.utc)
    live_start_date = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)

    syms = pd.read_csv('sp500.csv')
    syms = syms.values[:, 0].tolist()
    captial = 1000000

    agent = Agent(config, syms, captial)
    # agent.train(sd, ed)
    agent.test(sd, ed, live_start_date=live_start_date)

