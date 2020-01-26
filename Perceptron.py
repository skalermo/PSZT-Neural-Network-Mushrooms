import numpy as np


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoidDerivative(x):
    return x * (1 - x)


# Class definition
class Perceptron:
    def __init__(self, input_size, output_size, hidden_neurons=8, learning_rate=0.1, function=sigmoid, derivative=sigmoidDerivative):
        # Layers
        self.input = None
        self.hidden = None
        self.output = None

        # Weights
        self.weights1 = (np.random.rand(hidden_neurons, input_size) * 2 - 1) / np.sqrt(input_size)
        self.weights2 = np.zeros(shape=(output_size, hidden_neurons))

        # Learning rate
        self.lr = learning_rate

        # Biases
        self.hidden_biases = np.zeros(shape=(hidden_neurons, 1))
        self.output_biases = np.zeros(shape=(output_size, 1))

        # Activation function
        self.activation = function
        self.activation_derivative = derivative

    def feedforward(self):
        self.hidden = self.activation(np.dot(self.weights1, self.input) + self.hidden_biases)
        self.output = np.dot(self.weights2, self.hidden) + self.output_biases

    def backprop(self, y):
        d_weights2 = np.dot(2 * (y - self.output), self.hidden.T)
        d_weights1 = np.dot(np.dot(self.weights2.T, 2 * (y - self.output))
                            * self.activation_derivative(self.hidden), self.input.T)

        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2

    def train(self, x, y):
        self.input = x
        self.feedforward()
        self.backprop(y)

    def test(self, x):
        self.input = x
        self.feedforward()
        return self.getOutput()

    def getOutput(self):
        return self.output.reshape(self.output.shape[0])
