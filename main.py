from datetime import datetime
import tensorflow as tf
from config import AgentConfig
from dqn.agent import Agent
from dqn.environment import Environment

if __name__ == '__main__':
    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config, datafile_loc='./fundretriever/snp500.h5')

    agent = Agent(config, env)
    agent.train()

