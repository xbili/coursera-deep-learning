from nose.tools import eq_

from implementations.data_utils.minibatch_generator import minibatch_generator
from implementations.tests.utils.init_test_data import generate_classification_data


def test_minibatch_generator():
    """Test that we are generating our minibatch correctly."""

    m = 1000
    features = 10
    batch_size = 32
    X, Y = generate_classification_data(features, m)

    for idx, minibatch in enumerate(minibatch_generator(X, Y, batch_size)):
        minibatch_X, minibatch_Y = minibatch
        if idx != m // batch_size:
            eq_(minibatch_X.shape, (features, batch_size))
            eq_(minibatch_Y.shape, (1, batch_size))
        else:
            eq_(minibatch_X.shape, (features, m % batch_size))
            eq_(minibatch_Y.shape, (1, m % batch_size))
