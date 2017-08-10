class AgentConfig(object):

    # data parameters
    indicators = ['open', 'high', 'low', 'close', 'volume']

    # hyperparams
    memory_size = 100
    gamma = 0.9
    random_start = 30
    batch_size = 30
    learning_rate = 0.00025
    learning_rate_decay = 0.96
    n_epochs = 1000
    epsilon = .1
    C = 100
    n_history = 5
    tau = .001  # soft update for target network

    # network parameters
    n_hidden = 128
    n_layers = 10

    # checkpoints
    checkpoint_loc = '/tmp/dqn'
    resume_from_checkpoint = None