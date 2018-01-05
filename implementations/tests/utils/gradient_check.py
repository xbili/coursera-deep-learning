from copy import deepcopy

import numpy as np


def gradient_check(X, Y, model, epsilon=1e-7):
    """Performs a gradient check on a model that has passed through a single
    forward and back propagation.
    """

    approx, actual = [], []
    for key, param in model.params.items():
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                # Make copies of the model
                modelplus = deepcopy(model)
                modelminus = deepcopy(model)

                # Set the parameters
                modelplus.params[key][i][j] += epsilon
                modelminus.params[key][i][j] -= epsilon

                # Forward propagation
                modelplus._forward(X)
                modelminus._forward(X)

                # Compute loss
                AL_plus = modelplus.cache[f'A{modelplus.L}']
                AL_minus = modelminus.cache[f'A{modelminus.L}']

                loss_plus = modelplus.J(AL_plus, Y)
                loss_minus = modelminus.J(AL_minus, Y)

                # Calculate approximated gradient
                approx += [(loss_plus - loss_minus) / (2 * epsilon)]
                actual += [model.grads[f'd{key}'][i][j]]

                # Delete modelplus and modelminus to free up memory
                del modelplus
                del modelminus

    # Compute Euclidean distance
    approx, actual = np.array(approx), np.array(actual)
    diff_norms = np.linalg.norm(approx - actual)
    sum_of_norms = np.linalg.norm(approx) + np.linalg.norm(actual)
    euclidean = diff_norms / sum_of_norms

    if euclidean > epsilon:
        print(f'Euclidean: {euclidean}')
        return False

    return True
