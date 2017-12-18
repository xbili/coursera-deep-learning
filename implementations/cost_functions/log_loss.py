import numpy as np

def log_loss(AL, Y):
    """
    Computes the logistic loss of our predictions.
    """

    m = Y.shape[1]
    cost = (- 1 / m) * (np.dot(Y, np.log(AL.T)) +
                        np.dot(1 - Y, np.log(1 - AL.T)))

    # Ensure the single dimension cost value
    cost = np.squeeze(cost)

    return cost
