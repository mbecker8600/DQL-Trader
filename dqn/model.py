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


class Model:

    def __init__(self, state, reward, config, env):
        # network params
        self.n_outputs = 1
        self.n_inputs = env.get_num_stocks() * env.get_num_indicators() + env.get_num_stocks()
        self.n_hidden = config.n_hidden
        self.n_history = config.n_history

        # hyperparameters
        self.learning_rate = config.learning_rate

        # weights
        self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_outputs]), name='weights')
        self.biases = tf.Variable(tf.random_normal([self.n_outputs]), name='biases')

        # placeholders
        self.state = state
        self.reward = reward

        # methods
        self.prediction
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        X = tf.unstack(self.state, self.n_history, 1)
        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, reuse=None)
        outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        return tf.matmul(outputs[-1], self.weights) + self.biases

    @define_scope
    def optimize(self):
        loss = tf.reduce_mean(tf.squared_difference(self.reward, self.prediction))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

if __name__ == '__main__':
    from datetime import datetime
    from config import AgentConfig
    from dqn.environment import Environment
    sd = datetime(2005, 1, 1)
    ed = datetime(2015, 1, 1)
    config = AgentConfig()
    env = Environment(sd, ed, config)

    x = tf.placeholder("float", [None, 5, 3018])
    y = tf.placeholder("float", [None, 5, 5])
    model = Model(x, y, config, env)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    sess.run(model.prediction, {x: np.random.rand(1, 5, 3018)})
