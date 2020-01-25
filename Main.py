import numpy as np
import pandas as pd
from Perceptron import Perceptron

if __name__ == '__main__':
    # Read file
    data = pd.read_csv('dataset.csv')

    # Get output
    Y = data['class']

    # Get input
    X = data.drop('class', axis=1)
    X['ones'] = np.ones(X.shape[0])

    # Create NN
    input_size = X.shape[1]
    output_size = Y.shape[1] if len(Y.shape) > 1 else 1
    nn = Perceptron(input_size, output_size)

    # Convert to np arrays
    x = X.to_numpy(float)
    y = Y.to_numpy(float)

    # Add one more dimension to input array
    x.shape += (1,)

    for i in range(len(y)):
        if i % 100:
            print("for iteration # " + str(i) + "\n")
            print("Input : \n" + str(x[i].T))
            print("Actual Output: \n" + str(y[i]))
            print("Predicted Output: \n" + str(nn.output))
            print("Loss: \n" + str((np.square(y[i] - nn.output))))  # mean sum squared loss
            print("\n")

        nn.train(x[i], y[i])




