from implementations.models.fully_connected import FullyConnectedNN
from implementations.activations.sigmoid import Sigmoid
from implementations.activations.relu import ReLU
from implementations.cost_functions.log_loss import log_loss


def init_logistic_regression(features, examples):
    # Just 1 hidden layer, i.e. logistic regression
    hidden_layers = [(1, Sigmoid())]

    # Logistic loss
    loss = log_loss

    # Define model
    model = FullyConnectedNN(features, hidden_layers, loss)

    return model


def init_shallow_neural_network(features, examples):
    # 2 hidden layers
    hidden_layers = [(10, Sigmoid()), (1, Sigmoid())]

    # Logistic loss
    loss = log_loss

    # Define model
    model = FullyConnectedNN(features, hidden_layers, loss)

    return model


def init_deep_neural_network(features, examples):
    # 5 hidden layers
    hidden_layers = [
        (10, Sigmoid()),
        (10, Sigmoid()),
        (10, Sigmoid()),
        (10, Sigmoid()),
        (1, Sigmoid()),
    ]

    # Logistic loss
    loss = log_loss

    # Define model
    model = FullyConnectedNN(features, hidden_layers, loss)

    return model



