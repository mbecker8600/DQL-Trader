import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn import preprocessing


class Environment:
    def __init__(self, start_date, end_date, config, datafile_loc='../fundretriever/snp500.h5'):
        self.datafile_loc = datafile_loc
        self.date_range = self.__build_date_range__(start_date, end_date)
        self.sectors = self.__build_sectors__()
        self.num_stocks = self.__build_num_stocks__(self.sectors)
        self.num_indicators = 5
        self.n_history = config.n_history
        self.counter = self.n_history
        self.data = self.read_data_into_memory()

    def reset_environment(self):
        self.counter = self.n_history

    def get_date_range(self):
        return self.date_range

    def get_num_stocks(self):
        return self.num_stocks

    def get_num_indicators(self):
        return self.num_indicators

    def get_current_state(self):
        return preprocessing.normalize(np.reshape(self.get_state_at_timestep(self.counter).values, (1, self.num_indicators * self.num_stocks)))

    def get_n_history_state(self):
        state = []
        for t in range(self.counter - self.n_history, self.counter):
            s = self.get_state_at_timestep(t)
            state.append(s.values.ravel())
        return preprocessing.normalize(np.flip(np.reshape(np.array(state), (self.n_history, self.num_indicators * self.num_stocks)), axis=0))

    def read_data_into_memory(self):
        state = pd.Panel(major_axis=self.date_range, items=['close', 'high', 'low', 'open', 'volume'])
        for sector in self.sectors:
            close = pd.read_hdf(self.datafile_loc, sector)['close'].loc[self.date_range]
            high = pd.read_hdf(self.datafile_loc, sector)['high'].loc[self.date_range]
            low = pd.read_hdf(self.datafile_loc, sector)['low'].loc[self.date_range]
            open = pd.read_hdf(self.datafile_loc, sector)['open'].loc[self.date_range]
            volume = pd.read_hdf(self.datafile_loc, sector)['volume'].loc[self.date_range]
            data = pd.Panel({'close': close, 'high': high, 'low': low, 'open': open, 'volume': volume})
            state = pd.concat([state, data], axis=2)
        return state

    def get_state_at_timestep(self, timestep):
        return self.data.major_xs(self.date_range[timestep])

    def act(self, action):
        s = self.get_state_at_timestep(self.counter)  # get the current state
        self.counter += 1  # increment the counter by one to get the next state
        if self.counter >= len(self.date_range):  # determine if this is the terminal case
            terminal = True
            reward = 0.0
            s_prime = None
        else:
            terminal = False
            s_prime = self.get_current_state()  # get the next state
            reward = self.__calc_reward__(s, self.get_state_at_timestep(self.counter), action)  # calculate the reward for taking action a in state s

        return s_prime, reward, terminal

    def __calc_reward__(self, s, s_prime, action):
        perc_increase = (s_prime['close'] - s['close']) / s['close']
        allocations = perc_increase * action
        return allocations.sum()

    def __build_date_range__(self, sd, ed):
        spy = web.DataReader('SPY', 'google', sd, ed)
        return spy['Open'].index

    def __build_sectors__(self):
        store = pd.HDFStore(self.datafile_loc)
        return store.keys()

    def __build_num_stocks__(self, sectors):
        num_stocks = 0
        for sector in sectors:
            data = pd.read_hdf(self.datafile_loc, sector)
            num_stocks += data.shape[2]
        return num_stocks


if __name__ == '__main__':
    from datetime import datetime
    from config import AgentConfig

    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config)
    state = env.get_current_state()
    print(state)
    print(env.get_n_history_state())

    action = np.random.rand(env.num_stocks)
    action /= action.sum()
    s_prime, reward, terminal = env.act(action)
    print(reward)

