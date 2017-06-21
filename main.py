from datetime import datetime
from config import AgentConfig
from dqn.agent import Agent
from dqn.environment import Environment
import warnings


if __name__ == '__main__':
    warnings.simplefilter("ignore", DeprecationWarning)

    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config, datafile_loc='./fundretriever/snp500.h5')

    agent = Agent(config, env)
    agent.train()
    # agent.test(gen_plot=True, benchmark='VTI')

