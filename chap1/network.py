import random

import numpy as np


def _sigmoid(z):
    """Get the output value of the sigmoid function for a given z."""
    return 1.0 / (1.0 + np.exp(-z))


def _prime_sigmoid(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


class NeuralNetwork:

    def __init__(self, *sizes):
        """Init method for an artificial Neural Network."""
        self.num_layers = len(sizes)
        self._sizes = sizes
        # input layer does not have a bias
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(sizes[i], sizes[i + 1]) for i in range(0, len(sizes) - 1)]

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
            mini_batches = [training_data[k:k + mini_match_size] for k in range(0, n, mini_match_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)}/{n_test}.")
            else:
                print(f"Epoch {j}: complete.")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        pass

    def back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]



if __name__ == "__main__":
    nn = NeuralNetwork(2, 3, 1)
    print("hi")
