import numpy as np

from .neural_network import NeuralNetwork

class FullyConnectedNN(NeuralNetwork):
    """
    A vectorized implemention of a fully connected multi-layer perceptron
    neural network.
    """

    def __init__(self, input_shape, hidden_layers, loss, *args, **kwargs):
        """
        Initializes the neural network with initial weights.

        Parameters:
        - input_shape (tuple): shape of input matrix in the format:
            `(features, examples)` - note that training examples are stacked
            column-wise.
        - hidden_layers (List[tuples]): list of tuples representing the hidden
            layers in the format `(hidden units, activation)`.
        - loss (function): loss function to optimize.

        Returns: None.
        """

        super().__init__(*args, **kwargs)

        # Number of layers in neural network
        self.L = len(hidden_layers)

        # Loss function
        self.J = loss

        # Activation functions and units for each layer
        self.g = [lambda x: x] + [activation for _, activation in hidden_layers]
        self.n_x = [input_shape[0]] + [n_x for n_x, _ in hidden_layers]
        assert len(self.g) == len(self.n_x)

        # Weight matrix and bias vectors for each layer
        self.params = {}
        for l in range(1, self.L+1):
            self.params[f'W{l}'] = np.random.randn(self.n_x[l], self.n_x[l-1])
            self.params[f'b{l}'] = np.zeros(self.n_x[l]).reshape(-1, 1)

        # Cache to persist Z and A values for backpropagation
        self.cache = {}

        # Stores gradients for update
        self.grads = {}

    def __str__(self):
        """
        String representation of our model.
        """
        weights = []
        for name, w in self.params.items():
            weights += [f'{name} shape: {w.shape}']

        return '\n'.join(weights)


    def fit(self, X, y, batch_size=32, optimizer=None):
        """
        Trains the neural network with the specified optimization algorithm.
        """
        pass

    def predict(self, X, batch_size=32, optimizer=None):
        """
        Runs feedforward on a batch of data.
        """
        pass

    def evaluate(self, X, y, metric, batch_size=32):
        """
        Evaluates the performance of a neural network on a dataset.
        """
        pass

    def _forward(self, X):
        """
        One forward pass of data through the neural network, while storing
        essential information for backpropagation later.
        """

        # First input is simply our input data
        self.cache['A0'] = X

        for l in range(1, self.L+1):
            # Linear activation
            self.cache[f'Z{l}'] = np.dot(
                self.params[f'W{l}'],
                self.cache[f'A{l-1}']
            ) + self.params[f'b{l}']

            # Non-linear activation
            self.cache[f'A{l}'] = self.g[l].forward(self.cache[f'Z{l}'])

    def _backprop(self, X, Y):
        """
        One backward pass of data through the neural network, while storing
        the gradient updates required for the update step.
        """

        assert X.shape[1] == Y.shape[1]
        m = X.shape[1]

        # The input for the first step of backprop
        AL = self.cache[f'A{self.L}']
        self.grads[f'dA{self.L}'] = - (np.divide(Y, AL)
                                       - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(1, self.L+1)):
            self.grads[f'dZ{l}'] = self.grads[f'dA{l}']\
                * self.g[l].backward(self.cache[f'Z{l}'], self.grads[f'dA{l}'])
            self.grads[f'dW{l}'] = (1 / m)\
                * np.dot(self.grads[f'dZ{l}'], self.cache[f'A{l-1}'].T)
            self.grads[f'db{l}'] = (1 / m)\
                * np.sum(self.grads[f'dZ{l}'], axis=1, keepdims=True)
            self.grads[f'dA{l}'] = np.dot(
                self.params[f'W{l}'].T,
                self.grads[f'dZ{l}']
            )


    def _clear_cache(self):
        """
        Removes persisted values in cache and other banks.
        """
        self.cache = {}
        self.grads = {}
        self.params = {}
