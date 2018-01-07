import numpy as np


def generate_classification_data(features, examples=5):
    """Generates random input test data for classification tasks."""

    X = np.random.randn(features, examples)
    Y = np.random.randn(1, examples) < 0.5

    return X, Y
