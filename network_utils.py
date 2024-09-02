import numpy as np

__all__ = ("sigmoid", "sigmoid_prime")


def sigmoid(z):
    """Get the output value of the sigmoid function for a given z."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Get the output value of the sigmoid derivative with respect to z."""
    return sigmoid(z) * (1 - sigmoid(z))
