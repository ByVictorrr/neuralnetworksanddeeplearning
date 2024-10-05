from typing import Tuple, Callable
import pickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import softmax
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from mnist_loader import load_data
from theano.tensor.signal import pool


# Activation functions for neurons
def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)


from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

try_gpu = False
if try_gpu:
    print("Trying to run under a GPU.  If this is not desired, then modify network3.py to set the GPU flag to False.")
    try:
        theano.config.device = "gpu"
    except Exception:  # it's already set
        pass
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify network3.py to set the GPU flag to True.")


#### Load the MNIST data
def load_data_shared(filename="data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)


def size(data):
    """Return the size of the dataset data."""
    return data[0].get_value(borrow=True).shape[0]


def step_decay(epoch):
    initial_eta = 0.01
    drop = 0.5
    epochs_drop = 10
    return initial_eta * (drop ** np.floor((1 + epoch) / epochs_drop))


class Network:
    def __init__(self, layers, mini_batch_size):
        """Init for the Convolutional Neural Network.

        :param layers: a list of neuron layers
        :param mini_batch_size: to be used training by stochastic gradient descent
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        layers[0].set_inpt(self.x, self.x, mini_batch_size)
        for j in range(1, len(layers)):
            prev_layer, layer = layers[j - 1], layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, mini_batch_size)
        self.output = layers[-1].output
        self.output_dropout = layers[-1].output_dropout

    def sgd(self, training_data, epochs, validation_data, test_data, eta=step_decay, lmbda=0.0, patience=10):
        """Train the network using mini-batch stochastic gradient descent with early stopping and dynamic learning rate.

        eta can be a function of epoch (e.g., eta(epoch)).

        :param training_data: tuple of training data (x, y)
        :param epochs: number of epochs to train for
        :param eta: learning rate, can be a function of epoch (e.g., eta(epoch))
        :param validation_data: tuple of validation data (x, y)
        :param test_data: tuple of test data (x, y)
        :param lmbda: L2 regularization term
        :param patience: number of epochs with no improvement before stopping early
        """
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # Compute number of mini-batches for training, validation, and testing
        num_training_batches = size(training_data) // self.mini_batch_size
        num_validation_batches = size(validation_data) // self.mini_batch_size
        num_test_batches = size(test_data) // self.mini_batch_size

        # Define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)

        # Define functions to train a mini-batch and compute accuracy in validation and test mini-batches
        i = T.lscalar()  # mini-batch index

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
            }
        )

        # Initialize variables for training
        best_validation_accuracy = 0.0
        best_iteration = -1
        best_test_accuracy = 0.0
        patience_counter = 0  # To keep track of how many epochs without improvement

        for epoch in range(epochs):
            current_eta = eta(epoch)  # Compute the learning rate for the current epoch
            print(f"Epoch {epoch}, learning rate {current_eta}")

            # Update the weight update rule with the current learning rate
            updates = [(param, param - current_eta * grad) for param, grad in zip(self.params, grads)]

            # Define function to train a mini-batch with the updated learning rate
            train_mb_with_updates = theano.function(
                [i], cost, updates=updates,
                givens={
                    self.x: training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                    self.y: training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                }
            )

            # Perform training for the mini-batches
            for mini_batch_idx in range(int(num_training_batches)):
                iteration = num_training_batches * epoch + mini_batch_idx
                if iteration % 100 == 0:
                    print(f"Training mini-batch number {iteration}")
                cost_ij = train_mb_with_updates(mini_batch_idx)

            # After each epoch, check the validation accuracy
            validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(int(num_validation_batches))])
            print(f"Epoch {epoch}: validation accuracy {validation_accuracy:.2%}")

            # Check if the validation accuracy has improved
            if validation_accuracy > best_validation_accuracy:
                print("This is the best validation accuracy to date.")
                best_validation_accuracy = validation_accuracy
                best_iteration = epoch
                patience_counter = 0  # Reset patience counter since there is an improvement
                # Check test accuracy if test data is provided
                if test_data:
                    test_accuracy = np.mean([test_mb_accuracy(j) for j in range(int(num_test_batches))])
                    print(f"The corresponding test accuracy {test_accuracy:.2%}")
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience counter: {patience_counter}/{patience}")

            # Early stopping: stop training if no improvement for 'patience' epochs
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement in validation accuracy for {patience} epochs.")
                print(f"Best validation accuracy of {best_validation_accuracy:.2%} obtained at epoch {best_iteration}.")
                print(f"Corresponding test accuracy of {best_test_accuracy:.2%}")
                return  # Exit the function and stop training early

        print("Finished training network.")
        print(f"Best validation accuracy of {best_validation_accuracy:.2%} obtained at epoch {best_iteration}.")
        print(f"Corresponding test accuracy of {best_test_accuracy:.2%}")

    def accuracy(self, data):
        """Calculate accuracy for the given dataset.

        :param data: tuple containing input data and corresponding labels
        :return: the accuracy of the network's predictions on the data
        """
        x_data, y_data = data
        num_batches = size(data) // self.mini_batch_size
        i = T.lscalar()  # mini-batch index

        # Define a Theano function to compute accuracy for a mini-batch
        mini_batch_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x: x_data[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: y_data[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            }
        )

        # Calculate accuracy over all mini-batches
        accuracy = np.mean([mini_batch_accuracy(j) for j in range(num_batches)])
        return accuracy


class ConvPoolLayer:
    """Create a combination of a convolutional and a max-pooling layer.

        A more sophisticated implementation would separate the two, but for our purposes we will always use them
        together, and it simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape: Tuple[int, int, int, int], image_shape: Tuple[int, int, int, int],
                 pool_size=(2, 2), activation_fn: Callable = sigmoid):
        """Initialize the ConvPoolLayer.

        :param filter_shape: is a tuple of length 4.
                             1. the number of filters
                             2. the number of input feature maps
                             3. the filter height
                             4. the filter width
        :param image_shape: is a tuple of length 4.
                            1. mini_batch_size
                            2. number of input feature maps
                            3. the image height
                            4. the image width
        :param pool_size: is a tuple of length 2 - whose entries are the y and x pooling sizes
        :param activation_fn: activation function that each neuron will use in this layer
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation = activation_fn
        # initialize weights and biases
        n_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)
        self.w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                                          dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0])),
                                          dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]
        self.inpt = None
        self.output = None
        self.output_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Set the inputs to this layer."""
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ws=self.pool_size, ignore_border=True, mode="max")
        self.output = self.activation(pooled_out + self.b.dimshuffle("x", 0, "x", "x"))
        self.output_dropout = self.output


class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        """Fully connected layer init."""
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation_fn
        self.p_dropout = p_dropout
        # initialize weights and biases
        self.w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                                          dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(n_out,)),
                                          dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]
        self.inpt = None
        self.output = None
        self.y_out = None
        self.inpt_dropout = None
        self.output_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Set the inputs to this layer."""
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation(T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        """Get the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer:
    def __init__(self, n_in, n_out, p_dropout=0.0):
        """"""
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]
        # vars for set_inpt
        self.inpt = None
        self.output = None
        self.y_out = None
        self.inpt_dropout = None
        self.output_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """Set the inputs to this layer."""
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """Get the log-like cost."""
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        """Get the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))


if __name__ == "__main__":
    use_expand_data = False
    mini_batch_size = 10
    num_neuron_fully_layer = 100
    if use_expand_data:
        training_data, validation_data, test_data = load_data_shared("data/mnist_expanded.pkl.gz")
    else:
        training_data, validation_data, test_data = load_data_shared()
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      pool_size=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      pool_size=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40 * 4 * 4, n_out=num_neuron_fully_layer, activation_fn=ReLU),
        SoftmaxLayer(n_in=num_neuron_fully_layer, n_out=10)], mini_batch_size)
    net.sgd(training_data, 60, validation_data, test_data, eta=lambda epoch: 0.03, lmbda=0.1)


