import numpy as np

class ReLU(object):
    def __init__(self, rectified=0):
        self.rectified = rectified

    def forward(self, z):
        return np.maximum(self.rectified, z)

    def backward(self, z, dA):
        dZ = np.array(dA, copy=True)
        dZ[z < 0] = self.rectified

        assert dZ.shape == z.shape

        return dZ
