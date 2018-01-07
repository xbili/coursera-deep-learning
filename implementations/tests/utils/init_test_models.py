from implementations.models.fully_connected import FullyConnectedNN
from implementations.activations.sigmoid import Sigmoid
from implementations.cost_functions.log_loss import log_loss


def init_logistic_regression(features, examples):
    # 10 features, 50 training examples
    input_shape = (features, examples)

    # Just 1 hidden layer, i.e. logistic regression
    hidden_layers = [(1, Sigmoid())]

    # Logistic loss
    loss = log_loss

    # Define model
    model = FullyConnectedNN(input_shape, hidden_layers, loss)

    return model
