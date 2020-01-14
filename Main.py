import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    # Read file
    data = pd.read_csv('dataset.csv')

    # Get output
    Y = data['class']

    # Get input
    X = data.drop('class', axis=1)
    X['ones'] = np.ones(X.shape[0])

    # Create NN
    nn = NeuralNetwork(X, Y)

    # Convert to np arrays
    x = X.to_numpy(float)
    y = Y.to_numpy(float)

    for i in range(len(y)):
        if i % 100:
            print("for iteration # " + str(i) + "\n")
            print("Input : \n" + str(x[i]))
            print("Actual Output: \n" + str(y[i]))
            print("Predicted Output: \n" + str(nn.output))
            print("Loss: \n" + str((np.square(y[i] - nn.output))))  # mean sum squared loss
            print("\n")

        nn.train(x[i], y[i])




