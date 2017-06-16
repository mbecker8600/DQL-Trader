import numpy as np
from .replay_memory import ReplayMemory
from .q import Q
from tqdm import tqdm


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
        for epoch in tqdm(range(self.config.n_epochs)):
            self.env.reset_environment()  # reset environment each episode
            self.replay_memory.reset()  # reset replay memory
            print("Epoch {}".format(epoch))
            for timestep in tqdm(range(len(self.env.get_date_range()) - self.n_history)):
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
            if epoch % 50 == 0:  # save every 50 epochs
                self.Q.saver.save(self.Q.sess, "C:\\tmp\\dqn\\model.ckpt", global_step=epoch)


    def test(self):
        cum_reward = 0
        self.env.reset_environment()  # reset environment
        for timestep in tqdm(range(len(self.env.get_date_range()) - self.n_history)):
            s = self.env.get_current_state()
            action = self.Q.best_action(s)
            s_prime, reward, terminal = self.env.act(action)
            cum_reward += reward
        print(cum_reward)

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
