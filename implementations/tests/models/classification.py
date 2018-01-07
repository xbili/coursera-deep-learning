from nose.tools import eq_, ok_

from implementations.optimizers.gradient_descent import GradientDescent
from implementations.tests.utils.gradient_check import gradient_check
from implementations.tests.utils.init_test_models import init_logistic_regression
from implementations.tests.utils.init_test_data import generate_classification_data


FEATURES = 10
EXAMPLES = 50


def classification_suite(
    model_init,
    expected_L,
    expected_len_g,
    expected_len_n_x,
):
    """
    Runs the model through a simple series of test cases.

    Items tested:
    1. Model initialization
    2. Forward and backward propagation with gradient checking
    3. Simple gradient descent to ensure that loss is decreasing
    """

    X, Y = generate_classification_data(FEATURES, examples=EXAMPLES)

    model_init_test(model_init, X, Y, expected_L, expected_len_g, expected_len_n_x)
    model_forward_backprop_test(
        model_init, X, Y,
        expected_L,
        expected_len_g,
        expected_len_n_x,
    )
    model_gradient_descent_test(model_init, X, Y)

def model_init_test(
    model_init, X, Y,
    expected_L,
    expected_len_g,
    expected_len_n_x,
):
    model = model_init(FEATURES, EXAMPLES)

    model_invariant_test(model, expected_L, expected_len_g, expected_len_n_x)

    # Check parameter shape
    for l in range(1, model.L+1):
        actual_shape = model.params[f'W{l}'].shape
        expected_shape = (model.n_x[l], model.n_x[l-1])
        eq_(actual_shape, expected_shape,
            f'Model\'s actual shape should be {actual_shape}')

    # Banks should be empty
    ok_(not model.cache)
    ok_(not model.grads)


def model_forward_backprop_test(
    model_init, X, Y,
    expected_L,
    expected_len_g,
    expected_len_n_x,
):
    model = model_init(FEATURES, EXAMPLES)
    X, Y = generate_classification_data(FEATURES, examples=EXAMPLES)

    # Feedforward step
    model._forward(X)
    model_invariant_test(model, expected_L, expected_len_g, expected_len_n_x)

    # Check parameter shapes
    for l in range(1, model.L+1):
        actual_shape = model.params[f'W{l}'].shape
        expected_shape = (model.n_x[l], model.n_x[l-1])
        eq_(actual_shape, expected_shape,
            f'Param\'s actual shape should be {actual_shape}')

    eq_(len(model.cache), expected_L * 2 + 1)
    eq_(len(model.grads), 0) # Gradient should be empty, not calculated yet

    # Backprop
    model._backprop(X, Y)
    model_invariant_test(model, expected_L, expected_len_g, expected_len_n_x)

    # Perform a gradient check
    ok_(gradient_check(X, Y, model))


def model_gradient_descent_test(model_init, X, Y):
    model = model_init(FEATURES, EXAMPLES)
    X, Y = generate_classification_data(FEATURES, EXAMPLES)
    losses = model.fit(X, Y, epochs=20, optimizer=GradientDescent(learning_rate=0.01))

    prev = float('inf')
    for loss in losses:
        ok_(loss <= prev)


def model_invariant_test(model, expected_L, expected_len_g, expected_len_n_x):
    eq_(model.L, expected_L, f'Number of layers should be {expected_L}')
    eq_(len(model.g), expected_len_g,
        f'Number of activation functions should be {expected_len_g}')
    eq_(len(model.n_x), expected_len_n_x,
        f'Number of input shapes should be {expected_len_n_x}')
    eq_(len(model.g), len(model.n_x),
        'Number of input shapes should be equal to number of activations')
