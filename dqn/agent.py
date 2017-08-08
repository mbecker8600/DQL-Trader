import numpy as np
from .replay_memory import ReplayMemory
from .q import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pandas as pd


class Agent(object):

    def __init__(self, config, environment):
        self.env = environment
        self.config = config
        self.epsilon = config.epsilon
        self.copy_estimators = config.C
        self.replay_memory = ReplayMemory(self.config, environment)  # initialize replay memory
        self.Q = self.Q_hat = Q(config, environment, self.replay_memory)  # initialize Q functions
        self.n_history = config.n_history

    def train(self):
        start = 0 if self.config.resume_from_checkpoint is None else self.config.resume_from_checkpoint
        for epoch in tqdm(range(start, self.config.n_epochs)):
            self.env.reset_environment()  # reset environment each episode
            # self.replay_memory.reset()  # reset replay memory
            print("Epoch {}".format(epoch))
            for timestep in range(len(self.env.get_date_range()) - self.n_history):
                s = self.env.get_current_state()
                action = self.__choose_action__(s)
                s_prime, reward, terminal = self.env.act(action)
                self.replay_memory.store_transition(s, action, reward, s_prime)
                minibatch_samples = self.replay_memory.sample_replays()
                y = self.Q_hat.compute_targets(minibatch_samples)
                self.Q.train_critic_network(y, minibatch_samples)
                action_gradients = self.Q.get_action_gradients(minibatch_samples)
                self.Q.train_actor_network(action_gradients, minibatch_samples)
                if timestep % self.copy_estimators == 0:  # every timestep C, reset Q hat to Q
                    actor_weights = self.Q.get_actor_weights()
                    critic_weights = self.Q.get_critic_weights()
                    self.Q_hat.update_target_network(actor_weights, critic_weights)
            if epoch % 100 == 0:  # save every 10 epochs
                self.Q.saver.save(self.Q.sess, "C:\\tmp\\dqn\\model.ckpt", global_step=epoch)

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
