import numpy as np
from .replay_memory import ReplayMemory
from .portfolio_memory import PortfolioMemory
from .q import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pandas as pd
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, set_commission, set_slippage
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage


class Agent(object):
    def __init__(self, config, syms, capital=1000000):
        self.config = config
        self.epsilon = config.epsilon
        self.copy_estimators = config.C
        self.n_history = config.n_history
        self.syms = syms
        self.capital = capital
        self.replay_memory = None
        self.Q = self.Q_hat = None
        self.portfolio_memory = None

    def initialize_algo(self, context):
        set_commission(PerShare())
        set_slippage(VolumeShareSlippage())
        self.portfolio_memory = PortfolioMemory(len(self.syms), self.config.n_history)
        context.primed = False

    def handle_data(self, context, data):

        # initialize replay memory the first time
        if not self.replay_memory or not self.Q:
            data = data.history([symbol(sym) for sym in self.syms], self.config.indicators, self.config.n_history, '1d')
            state = self.__create_state__(data, self.portfolio_memory)
            self.replay_memory = ReplayMemory(self.config, state, len(self.syms))  # initialize replay memory
            self.Q = self.Q_hat = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions

        if not context.primed:  # run one iteration to start
            pass


        for sym in self.syms:
            pass
            # order_target_percent(symbol(sym), alloc)
        pass

    def train(self, sd, ed):
        start = 0 if self.config.resume_from_checkpoint is None else self.config.resume_from_checkpoint
        for epoch in tqdm(range(start, self.config.n_epochs)):
            print("Epoch {}".format(epoch))
            run_algorithm(initialize=self.initialize_algo,
                          handle_data=self.handle_data,
                          capital_base=1000000,
                          start=sd,
                          end=ed)











            # self.env.reset_environment()  # reset environment each episode
            # # self.replay_memory.reset()  # reset replay memory
            # print("Epoch {}".format(epoch))
            # for timestep in range(len(self.env.get_date_range()) - self.n_history):
            #     s = self.env.get_current_state()
            #     action = self.__choose_action__(s)
            #     s_prime, reward, terminal = self.env.act(action)
            #     self.replay_memory.store_transition(s, action, reward, s_prime)
            #     minibatch_samples = self.replay_memory.sample_replays()
            #     y = self.Q_hat.compute_targets(minibatch_samples)
            #     self.Q.train_critic_network(y, minibatch_samples)
            #     action_gradients = self.Q.get_action_gradients(minibatch_samples)
            #     self.Q.train_actor_network(action_gradients, minibatch_samples)
            #     if timestep % self.copy_estimators == 0:  # every timestep C, reset Q hat to Q
            #         actor_weights = self.Q.get_actor_weights()
            #         critic_weights = self.Q.get_critic_weights()
            #         self.Q_hat.update_target_network(actor_weights, critic_weights)
            # if epoch % 100 == 0:  # save every 10 epochs
            #     self.Q.saver.save(self.Q.sess, "C:\\tmp\\dqn\\model.ckpt", global_step=epoch)

    def test(self, gen_plot=False, benchmark='SPY'):
        returns = []
        self.env.reset_environment()  # reset environment
        for timestep in tqdm(range(len(self.env.get_date_range()) - self.n_history)):
            s = self.env.get_current_state()
            action = self.Q.best_action(s)
            s_prime, reward, terminal = self.env.act(action)
            returns.append(reward)
        returns = np.array(returns)
        print('Cumulative reward: {}'.format(returns.cumsum()[-1]))

        if gen_plot:
            date_range = self.env.get_date_range()
            benchmark_returns = web.DataReader(benchmark, 'google', date_range[0], date_range[-1])
            perc_change = (benchmark_returns['Close'] - benchmark_returns['Close'].shift(1)) / benchmark_returns['Close']

            returns_df = pd.DataFrame(columns=[benchmark, 'DQL'], index=date_range)
            returns_df[benchmark].iloc[1:] = perc_change
            returns_df['DQL'].iloc[len(date_range) - len(returns):] = returns
            returns_df.dropna(inplace=True)
            returns_df = returns_df.cumsum()
            returns_df.plot()
            plt.show()


    def __choose_action__(self, s):
        if self.__with_probability__(self.epsilon):  # with probability epsilon, return random action
            action = np.random.rand(self.env.num_stocks)
            action /= action.sum()
        else:
            action = self.Q.best_action(s)
        return action

    def __with_probability__(self, epsilon):
        random = np.random.random_sample()
        return True if random <= epsilon else False

    def __create_state__(self, data, portfolio_memory):
        state = np.reshape(np.swapaxes(data.values, 0, 1), (5, 20))
        state = np.concatenate((portfolio_memory.portfolio_allocations, state), axis=1)
        return state
