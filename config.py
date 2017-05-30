class AgentConfig(object):
    memory_size = 100
    gamma = 0.9
    random_start = 30
    batch_size = 3
    learning_rate = 0.00025
    learning_rate_decay = 0.96
    num_stocks = 500
    episode_size = 100
    epsilon = .1
    C = 10
    n_history = 5

    # network parameters
    n_hidden = 128
