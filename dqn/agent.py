import numpy as np
from .replay_memory import ReplayMemory
from .portfolio_memory import PortfolioMemory
from .q import Q
from tqdm import tqdm
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, set_commission, set_slippage, schedule_function
from zipline.utils.events import date_rules, time_rules
from zipline.finance.commission import PerShare
from zipline.finance.slippage import VolumeShareSlippage
import pyfolio as pf
from sklearn import preprocessing
from zipline.utils.tradingcalendar import trading_day
from zipline.data.benchmarks import get_benchmark_returns
from pandas import date_range
import tensorflow as tf
from sklearn import preprocessing


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
            self.timestep_progress = tqdm(total=len(trading_days) / 21)
            run_algorithm(initialize=self.initialize_training_algo,
                          capital_base=self.capital,
                          start=sd,
                          end=ed)

    def test(self, sd, ed, live_start_date, benchmark='SPY'):
        trading_days = date_range(sd, ed, freq=trading_day)
        self.timestep_progress = tqdm(total=len(trading_days) / 21)
        results = run_algorithm(initialize=self.initialize_testing_algo,
                                capital_base=self.capital,
                                start=sd,
                                end=ed)
        returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)
        benchmark_returns = get_benchmark_returns(benchmark, sd, ed)
        benchmark_returns.ix[sd] = 0.0
        pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                                  benchmark_rets=benchmark_returns, live_start_date=live_start_date, round_trips=True)

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
            with tf.variable_scope("Q"):
                self.Q = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions
            with tf.variable_scope("Q_hat"):
                self.Q_hat = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions

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
            y = preprocessing.scale(self.Q_hat.compute_targets(minibatch_samples))  # standardize the rewards.
            self.Q.train_critic_network(y, minibatch_samples)
            action_gradients = self.Q.get_action_gradients(minibatch_samples)
            self.Q.train_actor_network(action_gradients, minibatch_samples)
            if self.timestep % self.copy_estimators == 0:  # every timestep C, reset Q hat to Q
                actor_weights = self.Q.sess.run(self.Q.get_actor_weights())
                critic_weights = self.Q.sess.run(self.Q.get_critic_weights())
                self.Q_hat.update_target_network(actor_weights, critic_weights)
            if self.epoch % 10 == 0:  # save every 10 epochs
                self.Q.saver.save(self.Q.sess, "C:\\tmp\\dqn\\model.ckpt", global_step=self.epoch)

            # perform best action with newly trained network
            action = self.__choose_action__(state)
            context.previous_action = action
            self.__execute_orders__(data, action)

        context.previous_state = state
        self.timestep_progress.update(1)
        self.timestep += 1

    def testing_rebalance(self, context, data):
        historical_data = data.history(self.syms, self.config.indicators, self.config.n_history, '1d')
        state = self.__create_state__(historical_data, self.portfolio_memory)

        # initialize replay memory the first time
        if not self.replay_memory or not self.Q:
            self.replay_memory = ReplayMemory(self.config, state, len(self.syms))  # initialize replay memory
            self.Q = Q(self.config, self.replay_memory, len(self.syms))  # initialize Q functions

        action = self.Q.best_action(state)
        self.__execute_orders__(data, action)
        self.timestep_progress.update(1)

    def __validate_symbols__(self, syms):
        training_not_found = ['ANDV', 'BHGE', 'BHF', 'CSRA', 'DXC', 'FTV', 'HPE', 'HLT', 'INFO', 'KHC', 'PYPL', 'PPG',
                                'QRVO', 'SPGI', 'TRV', 'UA', 'WRK', 'WLTW', 'CPGX', 'BXLT', 'LIFE', 'MOLX', 'NYX', 'BMC',
                                'HNZ', 'CVH', 'PCS', 'TIE', 'CBE', 'GR', 'PGN', 'SLE', 'NVLS', 'EP', 'MHS', 'CEG', 'TLAB',
                                'WFR', 'CEPH', 'MI', 'MEE', 'NOVL', 'GENZ', 'MFE', 'AYE', 'KG', 'EK', 'SII', 'XTO', 'BJS',
                                'RX', 'SGP', 'CBE', 'ABK', 'TRB', 'DJ', 'AV', 'SBR', 'SBL', 'GLK', 'QTRN', 'BS']
        symbols = []
        print('symbols not found')
        for sym in syms:
            try:
                if sym not in training_not_found:  # HACK FOR TRAINING/TESTING DISCREPENCIES. NEED TO FIX
                    symbols.append(symbol(sym))
            except:
                print('{}'.format(sym))
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
