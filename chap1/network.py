import random

import numpy as np


class NeuralNetwork:

    def __init__(self, *sizes):
        """Init method for an artificial Neural Network."""
        self.num_layers = len(sizes)
        self._sizes = sizes
        # input layer does not have a bias
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(0, len(sizes) - 1)]

    def feed_forward(self, a):
        """Get the output of the network based on the input."""
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = 1 / (1 + np.exp(z))
        return a

    def sgd(self, training_data, epochs: int, mini_match_size: int, eta: float, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.

        :param training_data: list of tuples '(x, y)' representing the training inputs and the deserved outputs
        :param epochs: the number of epochs to train for
        :param mini_match_size: size of the mini-batches to use when sampling
        :param eta: the learning rate
        :param test_data: if not none the network will emulate against the test data after each epoch.
        """
        n_test = len(test_data) if test_data else 0
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_match_size] for k in range(0, n, mini_match_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)}/{n_test}.")
            else:
                print(f"Epoch {j}: complete.")



    def update_mini_batch(self, mini_batch, eta):
        pass

    def evaluate(self, test_data):
        pass


if __name__ == "__main__":
    nn = NeuralNetwork(2, 3, 1)
    print("hi")