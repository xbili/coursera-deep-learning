import numpy as np

from implementations.models.fully_connected import FullyConnectedNN
from implementations.activations.relu import ReLU
from implementations.activations.sigmoid import Sigmoid
from implementations.optimizers.gradient_descent import GradientDescent
from implementations.cost_functions.log_loss import log_loss

def load_data(path):
    """Loads the MNIST data from the specified path."""

    with open(path, 'rb') as f:
        mnist = np.load(f)

        X_train = mnist['x_train']
        Y_train = mnist['y_train']
        X_test = mnist['x_test']
        Y_test = mnist['y_test']

    return X_train, Y_train, X_test, Y_test


def flatten(X):
    """Flattens the 2D input shape"""
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def create_model(X_shape):
    features = X_shape[0]

    # 2 hidden layers
    hidden_layers = [(500, Sigmoid()), (1, Sigmoid())]

    # Logistic loss
    loss = log_loss

    # Define model
    return FullyConnectedNN(features, hidden_layers, loss)


def run():
    # Data from Keras downloaded source
    path = './implementations/examples/datasets/mnist.npz'

    X_train, Y_train, X_test, Y_test = load_data(path)

    X_train.astype('float32')
    X_test.astype('float32')

    # Flatten the 2D image
    X_train = flatten(X_train)
    X_test = flatten(X_test)

    X_train = X_train / 255
    X_test = X_test / 255

    # Stack the training examples column-wise instead
    X_train = X_train.T
    X_test = X_test.T

    # Reshape Y to make it consistent
    Y_train = Y_train.reshape(-1, 1).T
    Y_test = Y_test.reshape(-1, 1).T

    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')

    model = create_model(X_train.shape)

    learning_rate = 1e-3
    optimizer = GradientDescent(learning_rate=learning_rate)
    losses = model.fit(X_train, Y_train, optimizer=optimizer)

if __name__ == '__main__':
    run()
