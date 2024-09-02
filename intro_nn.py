import random

from mnist_loader import load_data_wrapper
from network import ArtificialNeuralNetwork
if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    hidden_neurons = random.choice([30, 100])
    print(f"Network using {hidden_neurons} for the hidden layer")
    nn = ArtificialNeuralNetwork(784, hidden_neurons, 10)
    nn.sgd(training_data, epochs=30, mini_batch_size=10, eta=3, test_data=test_data)
    print("hi")