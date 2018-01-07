from implementations.tests.utils.init_test_models import (
    init_logistic_regression,
    init_shallow_neural_network,
)
from implementations.tests.models.classification import classification_suite

def test_logistic_regression():
    """Simple logistic regression test for different models"""

    # Logistic Regression
    classification_suite(init_logistic_regression, 1, 2, 2)

    # Shallow neural network
    classification_suite(init_shallow_neural_network, 2, 3, 3)
