import numpy as np
from nose.tools import eq_, ok_

from implementations.models.fully_connected import FullyConnectedNN
from implementations.activations.sigmoid import Sigmoid
from implementations.cost_functions.log_loss import log_loss
from implementations.tests.utils.gradient_check import gradient_check
from implementations.tests.utils.init_test_models import init_logistic_regression
from implementations.tests.utils.init_test_data import generate_classification_data


FEATURES = 3
EXAMPLES = 5


def test_logistic_regression_init():
    """
    Test logistic regression initialization.

    Items tested:
    1. Number of layers in the model
    2. Shape of parameters
    3. Empty banks
    """
    # Define model
    model = init_logistic_regression(FEATURES, EXAMPLES)

    # Test assertions
    eq_(model.L, 1, 'Number of layers should be 1')
    eq_(len(model.g), 2, 'Number of activation functions should be 2')
    eq_(len(model.n_x), 2, 'Number of input shapes should be 2')
    eq_(len(model.g), len(model.n_x),
        'Number of input shapes should be equal to number of activations')

    # Check parameter shapes
    eq_(model.params['W1'].shape, (1, FEATURES))
    eq_(model.params['b1'].shape, (1, 1))

    # Banks should be empty
    ok_(not model.cache)
    ok_(not model.grads)


def test_logistic_regression_forward_backward_prop():
    """Tests a single forward and backward propagation iteration."""

    X, Y = generate_classification_data(FEATURES, examples=EXAMPLES)

    # Feedforward step
    model = init_logistic_regression(FEATURES, EXAMPLES)
    model._forward(X)

    # Test assertions for model invariants
    eq_(model.L, 1, 'Number of layers should be 1')
    eq_(len(model.g), 2, 'Number of activation functions should be 2')
    eq_(len(model.n_x), 2, 'Number of input shapes should be 2')
    eq_(len(model.g), len(model.n_x),
        'Number of input shapes should be equal to number of activations')

    # Check parameter shapes
    eq_(model.params['W1'].shape, (1, FEATURES))
    eq_(model.params['b1'].shape, (1, 1))

    # Banks should now be filled with A0, Z1, A1
    eq_(len(model.cache), 3)

    # Gradients should still be empty, not calculated yet.
    eq_(len(model.grads), 0)

    # Backprop
    model._backprop(X, Y)

    # Test assertions for model invariants
    eq_(model.L, 1, 'Number of layers should be 1')
    eq_(len(model.g), 2, 'Number of activation functions should be 2')
    eq_(len(model.n_x), 2, 'Number of input shapes should be 2')
    eq_(len(model.g), len(model.n_x),
        'Number of input shapes should be equal to number of activations')

    # Check parameter shapes
    eq_(model.params['W1'].shape, (1, FEATURES))
    eq_(model.params['b1'].shape, (1, 1))

    # Check gradient shapes
    eq_(model.grads['dW1'].shape, (1, FEATURES))
    eq_(model.grads['db1'].shape, (1, 1))

    # Perform a gradient check
    ok_(gradient_check(X, Y, model))
