import unittest
import numpy as np


import Perceptron


class TestPerceptron(unittest.TestCase):
    def test_initialDimensions(self):
        input_size = 50
        output_size = 4
        neurons_count = 4
        p = Perceptron.Perceptron(input_size, output_size, neurons_count)

        self.assertEqual((neurons_count, input_size), p.weights1.shape)
        self.assertEqual((output_size, neurons_count), p.weights2.shape)
        self.assertEqual((neurons_count, 1), p.hidden_biases.shape)
        self.assertEqual((output_size, 1), p.output_biases.shape)

    def test_feedforwardDimensions(self):
        input_size = 50
        output_size = 4
        neurons_count = 4
        p = Perceptron.Perceptron(input_size, output_size, neurons_count)
        input_data = list(range(input_size))
        p.input = np.array(input_data)
        p.input.shape += (1, )

        self.assertEqual((input_size, 1), p.input.shape)

        p.feedforward()
        self.assertEqual((neurons_count, 1), p.hidden.shape)
        self.assertEqual((output_size, 1), p.output.shape)

    # def test_trainDimensions(self):
    #     input_size = 50
    #     output_size = 4
    #     neurons_count = 4
    #     p = Perceptron.Perceptron(input_size, output_size, neurons_count)
    #     input_data = list(range(input_size))
    #     output_data = np.ones((output_size,))
    #     p.input = np.array(input_data)
    #     p.input.shape += (1,)
    #     p.feedforward()


if __name__ == '__main__':
    unittest.main()

