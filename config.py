class AgentConfig(object):

    # data parameters
    indicators = ['open', 'high', 'low', 'close', 'volume']

    # hyperparams
    memory_size = 1000
    gamma = 0.8
    random_start = 30
    batch_size = 50
    learning_rate = 0.00025
    learning_rate_decay = 0.96
    n_epochs = 10000
    epsilon = .1
    C = 50  # update the true network every C timesteps
    n_history = 30

    # network parameters
    n_hidden = 128
    n_layers = 10

    # checkpoints
    checkpoint_loc = '/tmp/dqn'
    resume_from_checkpoint = 219
