import functools
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class ActorModel:

    def __init__(self, state, action_gradient, config, n_syms):
        # network params
        self.n_outputs = n_syms * 2
        self.n_inputs = n_syms + n_syms * len(config.indicators)
        self.n_hidden = config.n_hidden
        self.n_history = config.n_history

        # hyperparameters
        self.learning_rate = config.learning_rate

        # weights
        self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_outputs]), name='actor_weights')
        self.biases = tf.Variable(tf.random_normal([self.n_outputs]), name='actor_biases')

        # placeholders
        self.state = state
        self.action_gradient = action_gradient  # This gradient will be provided by the critic network

        self.actor_gradients = tf.gradients(self.prediction, [self.weights, self.biases], -self.action_gradient)

        # methods
        self.prediction
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        X = tf.unstack(self.state, self.n_history, 1)
        with tf.variable_scope('actor_model'):
            lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, reuse=None, activation=tf.nn.relu)
            outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        return tf.matmul(outputs[-1], self.weights) + self.biases

    @define_scope
    def optimize(self):
        return tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, [self.weights, self.biases]))

if __name__ == '__main__':
    from datetime import datetime
    from config import AgentConfig
    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    n_sym = 4

    x = tf.placeholder("float", [None, config.n_history, n_sym * len(config.indicators)])
    y = tf.placeholder("float", [None, config.n_history, n_sym])
    model = ActorModel(x, y, config, n_sym)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    sess.run(model.prediction, {x: np.random.rand(1, config.n_history, n_sym * len(config.indicators))})
