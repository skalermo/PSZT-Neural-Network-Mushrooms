import numpy as np


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoidDerivative(x):
    return x * (1 - x)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y, hidden_neurons=8, function=sigmoid, derivative=sigmoidDerivative):
        # Layers
        self.input = None
        self.hidden = None
        self.output = None

        output_size = y.shape[1] if len(y.shape) > 1 else 1

        # Weights
        self.weights1 = np.random.rand(x.shape[1], hidden_neurons)
        self.weights2 = np.random.rand(hidden_neurons, output_size)
        
        # Activation function
        self.activation = function
        self.derivative = derivative

    def feedforward(self):
        self.hidden = self.activation(np.dot(self.input, self.weights1))
        self.output = self.activation(np.dot(self.hidden, self.weights2))

    def backprop(self, y):
        d_weights2 = np.dot(self.hidden.T, 2 * (y - self.output) * self.derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (y - self.output) * self.derivative(self.output),
                                                 self.weights2.T) * self.derivative(self.hidden))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, x, y):
        self.input = x
        self.feedforward()
        self.backprop(y)