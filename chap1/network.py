import numpy as np


class NeuralNetwork:

    def __init__(self, *sizes):
        self.num_layers = len(sizes)
        self._sizes = sizes
        # input layer does not have a bias
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(0, len(sizes) - 1)]

if __name__ == "__main__":
    nn = NeuralNetwork(2, 3, 1)
    print("hi")