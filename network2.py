"""An imported version of network.py

Implementing the stochastic gradient descent learning als for a feedforward neural network.

Improvements:
1. addition of the cross-entropy cost function
2. addition of regularization
3. better initialization of network weights.
"""
import abc
import json
from typing import Optional
import random
import numpy as np

from network_utils import sigmoid, sigmoid_prime


class NeuralNetworkCost(abc.ABC):

    @abc.abstractmethod
    def cost(self, a, y):
        """Get the output of the cost function."""
        pass

    @abc.abstractmethod
    def delta(self, z, a, y):
        """Get the error from the last layer."""
        pass


class QuadraticCost(NeuralNetworkCost):
    def cost(self, a, y):
        """Get the cost associated with an output 'a' and desired output 'y'."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    def delta(self, z, a, y):
        """Get the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(NeuralNetworkCost):

    def cost(self, a, y):
        """Get the cost associated with an output 'a' and desired output 'y'."""
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def delta(self, z, a, y):
        """Get the error delta from the output layer."""
        return a - y


class Network:

    def __init__(self, *sizes, cost: Optional[NeuralNetworkCost] = None):
        """Init method for an artificial Neural Network."""
        self.num_layers = len(sizes)
        self._sizes = sizes
        self._cost_obj = cost if cost else CrossEntropyCost()
        self.weights, self.biases, self.velocities = [], [], []
        self.biases = [np.random.randn(s, 1) for s in self._sizes[1:]]
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0 and std_dev 1/sqrt(n_in)."""
        self.biases = [np.random.randn(s, 1) for s in self._sizes[1:]]
        self.weights = [np.random.randn(self._sizes[i + 1], self._sizes[i]) / np.sqrt(self._sizes[i]) for i in
                        range(0, len(self._sizes) - 1)]
        # Momentum-based
        self.velocities = [np.random.randn(self._sizes[i + 1], self._sizes[i]) for i in range(0, len(self._sizes) - 1)]

    def large_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0 and std_dev 1."""
        self.biases = [np.random.randn(s, 1) for s in self._sizes[1:]]
        self.weights = [np.random.randn(self._sizes[i + 1], self._sizes[i]) for i in range(0, len(self._sizes) - 1)]
        # Momentum-based
        self.velocities = [np.random.randn(self._sizes[i + 1], self._sizes[i]) for i in range(0, len(self._sizes) - 1)]

    def feed_forward(self, a):
        """Get the output of the network based on the input."""
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, friction_coef=.5,
            patience=10, use_learning_schedule=True,
            evaluation_data=None, **kwargs):
        monitor_evaluation_cost = kwargs.pop("monitor_training_cost", False)
        monitor_evaluation_accuracy = kwargs.pop("monitor_evaluation_accuracy", False)
        monitor_training_cost = kwargs.pop("monitor_training_cost", False)
        monitor_training_accuracy = kwargs.pop("monitor_training_accuracy", False)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        best_accuracy = 0  # To keep track of the best accuracy so far
        patience_counter = 0  # To keep track of how many epochs without improvement
        initial_eta = eta  # Store the initial learning rate
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, friction_coef, len(training_data))
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, is_training_data=True)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, is_training_data=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {(accuracy/len(training_data))*100=}")
                # Check if current evaluation accuracy is better than the best seen so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0  # Reset patience counter if improvement is seen
                else:
                    patience_counter += 1

                # Check if patience is exhausted
                if patience_counter >= patience:
                    if use_learning_schedule:
                        eta /= 2  # Halve the learning rate
                        patience_counter = 0  # Reset the patience counter
                        print(f"No improvement in {patience} epochs, halving learning rate to {eta}")
                        # Check if learning rate has dropped below 1/128 of the initial value
                        if eta < initial_eta / 128:
                            print(f"Learning rate has dropped below 1/128 of the initial value, stopping training.")
                            break
                        print(f"Learning rate is now: {eta=} from {initial_eta=}")
                    else:
                        print(f"Stopping early at epoch {j + 1} due to no improvement in {patience} epochs.")
                        break

            if monitor_evaluation_cost and evaluation_data:
                cost = self.total_cost(evaluation_data, is_training_data=False)
                training_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data, is_training_data=False)
                training_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy/len(evaluation_data)*100=}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmda, friction_coef, training_size):
        """Update nn weighs & biases by applying gradient descent using backpropagation to a single mini batch.

        :param mini_batch: a list of tuples
        :param eta: is the learning rate
        :param lmda: the regularization parameter
        :param training_size: size of the full training set
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

        self.velocities = [
            friction_coef * v - (eta / len(mini_batch)) * nw
            for v, nw in zip(self.velocities, nabla_w)
        ]
        # Update weights using velocities (momentum term)
        self.weights = [
            (1 - (eta * lmda) / training_size) * w + v
            for w, v in zip(self.weights, self.velocities)
        ]
        #  b^l=b^l - (eta/m) * sum(delta^{x, l})
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
            activation = sigmoid(z)
            activations.append(activation)
        # Step 3: Output error direct_delta^{x,L}: Compute the vector
        #   diract_deta^{x,L} = grad_a(C_x) (*) sigma_prime(z^{x, L})
        delta = self._cost_obj.delta(zs[-1], activations[-1], y)
        # dC/db^L = delta^L
        nabla_b[-1] = delta
        # dC/dw^L = delta^L * a^{L-1}
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Step 4: Back-propagate the error:
        #   for each l in [L-1, L-2, ..., 2]
        #       delta^{x, l} = ((w^{l+1})^T * delta^{l+1}) (*) sigma_prime(z^l)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, is_training_data=False):
        """Get the number of inputs in data for which the nn outputs the correct result."""
        if is_training_data:
            # np.argmax - get the index with the highest value
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y) for x, y in data]
        return sum(int(predicted == actual) for predicted, actual in results)

    def total_cost(self, data, is_training_data=False):
        """Get the total cost of the trained network."""
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if not is_training_data:
                e = np.zeros((10, 1))
                e[y] = 1
                y = e
            cost += self._cost_obj.cost(a, y) / len(data)
        return cost

    def save(self, file_path):
        """Save the Neural Network to a file."""
        with open(file_path, "w") as flp:
            json.dump({
                "sizes": list(self._sizes),
                "weights": [w.tolist() for w in self.weights],
                "biases": [w.tolist() for w in self.weights],
                "cost": str(self._cost_obj.__class__.__name__)
            }, flp)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r") as flp:
            data = json.load(flp)
        cost = globals()[data["cost"]]()
        net = cls(*data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net


if __name__ == "__main__":
    from mnist_loader import load_data_wrapper

    training_data, validation_data, test_data = load_data_wrapper()
    nn = Network(784, 30, 10)
    epochs = 30
    eta = 0.5
    mini_batch_size = 30
    nn.sgd(training_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta, lmbda=5.0,
           evaluation_data=validation_data,
           monitor_evaluation_accuracy=True,
           monitor_evaluation_cost=True,
           monitor_training_accuracy=True,
           monitor_training_cost=True,
           )
    nn.save("enhanced_neural_network.json")
    loaded_nn = Network.load("enhanced_neural_network.json")
    print('hi')
