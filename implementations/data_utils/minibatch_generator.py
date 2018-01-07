import numpy as np


def minibatch_generator(X, Y, batch_size, shuffle=True):
    """Yields minibatches of data given input data and its labels."""

    m = X.shape[1]

    # TODO: Shuffle X and Y data

    for i in range(0, m, batch_size):
        yield X[:, i:i+batch_size], Y[:, i:i+batch_size]
