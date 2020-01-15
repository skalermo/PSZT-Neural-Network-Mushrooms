import numpy as np


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoidDerivative(x):
    return x * (1 - x)


# Class definition
class Perceptron:
    def __init__(self, input_size, output_size, hidden_neurons=8, function=sigmoid, derivative=sigmoidDerivative):
        # Layers
        self.input = None
        self.hidden = None
        self.output = None

        # Weights
        self.weights1 = np.random.rand(hidden_neurons, input_size)
        self.weights2 = np.random.rand(output_size, hidden_neurons)
        
        # Activation function
        self.activation = function
        self.derivative = derivative

    def feedforward(self):
        self.hidden = self.activation(np.dot(self.weights1, self.input))
        self.output = self.activation(np.dot(self.weights2, self.hidden))

    def backprop(self, y):
        d_weights2 = np.dot(2 * (y - self.output) * self.derivative(self.output), self.hidden.T)
        d_weights1 = np.dot(np.dot(self.weights2.T, 2 * (y - self.output) * self.derivative(self.output))
                            * self.derivative(self.hidden), self.input.T)

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, x, y):
        self.input = x
        self.feedforward()
        self.backprop(y)
