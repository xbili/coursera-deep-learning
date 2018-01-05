import numpy as np

class Sigmoid(object):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z, dA):
        s = self.forward(z)
        dZ = s * (1-s)
        assert dZ.shape == z.shape
        return dZ
