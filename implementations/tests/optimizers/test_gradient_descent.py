import numpy as np
from nose.tools import ok_

from implementations.optimizers.gradient_descent import GradientDescent
from implementations.tests.utils.init_test_models import init_logistic_regression
from implementations.tests.utils.init_test_data import generate_classification_data

def test_gradient_descent():
    """Performs a simple test for a gradient descent update."""

    features = 5
    examples = 5
    X, Y = generate_classification_data(features, examples=examples)
    model = init_logistic_regression(features, examples)
    params_before = model.params

    learning_rate = 1e-4
    sut = GradientDescent(learning_rate=learning_rate)

    model._forward(X)
    model._backprop(X, Y)
    grads = model.grads

    expected = {}
    for key, param in params_before.items():
        expected[key] = param - learning_rate * grads[f'd{key}']

    res = sut.update(model.params, model.grads)

    for key in model.params:
        ok_(np.equal(res[key], expected[key]).all(),
            'Params after update should be equal to the expected result.')
