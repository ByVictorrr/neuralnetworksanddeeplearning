import random

import numpy as np


def _sigmoid(z):
    """Get the output value of the sigmoid function for a given z."""
    return 1.0 / (1.0 + np.exp(-z))


def _prime_sigmoid(z):
    return _sigmoid(z) * (1 - _sigmoid(z))


class ArtificialNeuralNetwork:

    def __init__(self, *sizes):
        """Init method for an artificial Neural Network."""
        self.num_layers = len(sizes)
        self._sizes = sizes
        # input layer does not have a bias
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(sizes[i + 1], sizes[i]) for i in range(0, len(sizes) - 1)]

    def feed_forward(self, a):
        """Get the output of the network based on the input."""
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = _sigmoid(z)
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

    def evaluate(self, test_data):
        """Get the number of test inputs for which the nn outputs the correct result."""
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def update_mini_batch(self, mini_batch, eta):
        """Update nn weighs & biases by applying gradient descent using backpropagation to a single mini batch.

        :param mini_batch: a list of tuples
        :param eta: is the learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Gradient descent:
        #   for l in [L, L-1, ...2]:
        for x, y in mini_batch:
            # delta_nabla_b = delta^{x, l}
            # delta_nabla_w = delta^{x,l} * (a^{x, l-1})^T
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            # nabla_b = sum(delta^{x, l})
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # nabla_w = sum(delta^{x,l} * (a^{x, l-1})^T)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #  w^l=w^l - (eta/m) * sum(delta^{x,l} * (a^{x, l-1})^T)
        #  b^l=b^l - (eta/m) * sum(delta^{x, l})
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        """Get a tuple representing the gradient for the cost function C_x."""
        # nabla_b and nabla_w are layer-by-layer lists
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Step 1: Set the corresponding input activation a^{x, 1}
        activation = x
        # List to store all activations, layer by layer
        activations = [x]
        # List to store all the z vectors, Layer by layer
        zs = []
        # Step 2: Feedforward:
        #   for each l in [2, 3,..., L]
        #       z^{x,l} = w^l*a^{x, l-1} + b^l
        #       a^{x, l} = sigma(z^{x, l})
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            # the output from layer i
            activation = _sigmoid(z)
            activations.append(activation)
        # Step 3: Output error direct_delta^{x,L}: Compute the vector
        #   diract_deta^{x,L} = grad_a(C_x) (*) sigma_prime(z^{x, L})
        delta = (activations[-1] - y) * _prime_sigmoid(zs[-1])
        # dC/db^L = delta^L
        nabla_b[-1] = delta
        # dC/dw^L = delta^L * a^{L-1}
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Step 4: Back-propagate the error:
        #   for each l in [L-1, L-2, ..., 2]
        #       delta^{x, l} = ((w^{l+1})^T * delta^{l+1}) (*) sigma_prime(z^l)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = _prime_sigmoid(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w


if __name__ == "__main__":
    nn = ArtificialNeuralNetwork(2, 3, 1)
    print("hi")
