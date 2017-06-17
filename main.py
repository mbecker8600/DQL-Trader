from datetime import datetime
from config import AgentConfig
from dqn.agent import Agent
from dqn.environment import Environment

if __name__ == '__main__':
    sd = datetime(2014, 12, 20)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config, datafile_loc='./fundretriever/snp500.h5')

    agent = Agent(config, env)
    agent.train()
    # agent.test()

