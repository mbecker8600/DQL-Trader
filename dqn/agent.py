import numpy as np
from .replay_memory import ReplayMemory
from .portfolio_memory import PortfolioMemory
from .q import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pandas as pd
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, set_commission, set_slippage, schedule_function
from zipline.utils.events import date_rules, time_rules
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage
from sklearn import preprocessing
from zipline.utils.tradingcalendar import trading_day
from pandas import date_range


class Agent(object):
    def __init__(self, config, syms, capital=1000000):
        self.config = config
        self.epsilon = config.epsilon
        self.copy_estimators = config.C
        self.n_history = config.n_history
        self.ticker_syms = syms
        self.syms = None
        self.capital = capital
        self.replay_memory = None
        self.Q = self.Q_hat = None
        self.portfolio_memory = None
        self.timestep = 0

    def train(self, sd, ed):
        start = 0 if self.config.resume_from_checkpoint is None else self.config.resume_from_checkpoint
        for self.epoch in tqdm(range(start, self.config.n_epochs)):
            print("Epoch {}".format(self.epoch))
            trading_days = date_range(sd, ed, freq=trading_day)
            self.timestep_progress = tqdm(total=len(trading_days) / 12)
            run_algorithm(initialize=self.initialize_training_algo,
                          capital_base=self.capital,
                          start=sd,
                          end=ed)

    def test(self, sd, ed, gen_plot=False, benchmark='SPY'):
        trading_days = date_range(sd, ed, freq=trading_day)
        self.timestep_progress = tqdm(total=len(trading_days) / 12)
        results = run_algorithm(initialize=self.initialize_testing_algo,
                                capital_base=self.capital,
                                start=sd,
                                end=ed)
        results.portfolio_value.plot()
        plt.show()

    def initialize_training_algo(self, context):
        set_commission(PerShare())
        set_slippage(VolumeShareSlippage())
        if self.syms is None:
            self.syms = self.__validate_symbols__(self.ticker_syms)
        self.portfolio_memory = PortfolioMemory(len(self.syms), self.config.n_history)
        schedule_function(self.training_rebalance,
                          date_rule=date_rules.month_start(),
                          time_rule=time_rules.market_open(minutes=1))
        context.primed = False

    def initialize_testing_algo(self, context):
        set_commission(PerShare())
        set_slippage(VolumeShareSlippage())
        if self.syms is None:
            self.syms = self.__validate_symbols__(self.ticker_syms)
        self.portfolio_memory = PortfolioMemory(len(self.syms), self.config.n_history)
        schedule_function(self.testing_rebalance,
                          date_rule=date_rules.month_start(),
                          time_rule=time_rules.market_open(minutes=1))

    def training_rebalance(self, context, data):
        # initial setup
        if context.primed:
            weights = self.__get_current_portfolio_weights__(context)
            self.portfolio_memory.add_memory(weights)
        historical_data = data.history(self.syms, self.config.indicators, self.config.n_history, '1d')
        state = self.__create_state__(historical_data, self.portfolio_memory)

        # initialize replay memory the first time
        if not self.replay_memory or not self.Q:
            self.replay_memory = ReplayMemory(self.config, state, len(self.syms))  # initialize replay memory
            self.Q = self.Q_hat = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions
            # self.Q_hat = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q_hat functions

        if not context.primed:  # run one iteration to start
            action = self.__choose_action__(state)
            context.previous_action = action
            self.__execute_orders__(data, action)
            context.primed = True  # set to true after first run
        else:
            # calculate reward and train network from previous action result
            reward = context.portfolio.portfolio_value
            self.replay_memory.store_transition(context.previous_state[0], context.previous_action, reward, state[0])
            minibatch_samples = self.replay_memory.sample_replays()
            y = self.Q_hat.compute_targets(minibatch_samples)
            self.Q.train_critic_network(y, minibatch_samples)
            action_gradients = self.Q.get_action_gradients(minibatch_samples)
            self.Q.train_actor_network(action_gradients, minibatch_samples)
            if self.timestep % self.copy_estimators == 0:  # every timestep C, reset Q hat to Q
                actor_weights = self.Q.get_actor_weights()
                critic_weights = self.Q.get_critic_weights()
                self.Q_hat.update_target_network(actor_weights, critic_weights)
            if self.epoch % 10 == 0:  # save every 10 epochs
                self.Q.saver.save(self.Q.sess, "C:\\tmp\\dqn\\model.ckpt", global_step=self.epoch)

            # perform best action with newly trained network
            action = self.__choose_action__(state)
            context.previous_action = action
            self.__execute_orders__(data, action)

        context.previous_state = state
        self.timestep_progress.update(1)

    def testing_rebalance(self, context, data):
        historical_data = data.history(self.syms, self.config.indicators, self.config.n_history, '1d')
        state = self.__create_state__(historical_data, self.portfolio_memory)

        # initialize replay memory the first time
        if not self.replay_memory or not self.Q:
            self.replay_memory = ReplayMemory(self.config, state, len(self.syms))  # initialize replay memory
            self.Q = self.Q_hat = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions

        action = self.__choose_action__(state)
        self.__execute_orders__(data, action)
        self.timestep_progress.update(1)

    def __validate_symbols__(self, syms):
        symbols = []
        for sym in syms:
            try:
                symbols.append(symbol(sym))
            except:
                print('Symbol {} not found. Not training on it.'.format(sym))
        return symbols

    def __choose_action__(self, s):
        if self.__with_probability__(self.epsilon):  # with probability epsilon, return random action
            action = np.random.rand(len(self.syms))
            action /= action.sum()
        else:
            action = self.Q.best_action(s)
        return action

    def __with_probability__(self, epsilon):
        random = np.random.random_sample()
        return True if random <= epsilon else False

    def __create_state__(self, data, portfolio_memory):
        data = np.nan_to_num(data)
        state = np.reshape(np.swapaxes(data, 0, 1), (self.n_history, len(self.syms) * len(self.config.indicators)))
        state = np.concatenate((portfolio_memory.portfolio_allocations, state), axis=1)
        return preprocessing.normalize(state)

    def __execute_orders__(self, data, action):
        for sym, target_allocation in zip(self.syms, action):
            if data.can_trade(sym):
                order_target_percent(sym, target_allocation)

    def __get_current_portfolio_weights__(self, context):
        allocations = []
        for sym in self.syms:
            try:
                allocations.append(context.portfolio.current_portfolio_weights[sym])
            except IndexError:
                allocations.append(0.0)
        return np.array(allocations)
