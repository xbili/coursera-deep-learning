class GradientDescent(object):
    """Classic Gradient Descent parameter updates."""

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """Performs a single update step of the optimization algorithm."""
        for key, param in params.items():
            params[key] = param - self.learning_rate * grads[f'd{key}']

        return params
