from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    """Base abstract class that all neural network models should implement"""
    @abstractmethod
    def fit(self, X, y, batch_size=None, optimizer=None):
        pass

    @abstractmethod
    def predict(self, X, batch_size=None, optimizer=None):
        pass
