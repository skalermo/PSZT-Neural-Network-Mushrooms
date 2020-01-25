import numpy as np
import pandas as pd
from Perceptron import Perceptron
from Chart import Chart
from KFold import KFold


def readData(filename):
    # Read file
    data = pd.read_csv(filename)

    # Get output
    y = data['class']

    # Get input
    x = data.drop('class', axis=1)

    # Convert to np arrays
    x = x.to_numpy(float)
    y = y.to_numpy(float)

    # Add one more dimension to input array
    x.shape += (1,)
    y.shape += (1,)

    return x, y


if __name__ == '__main__':

    input, output = readData('dataset.csv')
    # Create NN
    p = Perceptron(input.shape[1], output.shape[1], hidden_neurons=16)

    k = 5

    kfold = KFold(k, True)

    train_idx, test_idx = next(kfold.split(input))
    train_input = input[train_idx]
    train_output = output[train_idx]
    test_input = input[test_idx]
    test_output = output[test_idx]

    train_loss = []
    test_loss = []

    for j in range(100):
        loss_sum = 0.0
        for i in range(len(train_output)):
            # if i == len(train_output)-1:
            #     print("for iteration # " + str(i) + "\n")
            #     print("Actual Output: \n" + str(train_output[i]))
            #     print("Predicted Output: \n" + str(p.getOutput()))
            #     print("Loss: \n" + str((np.square(train_output[i] - p.getOutput()))))  # mean sum squared loss
            #     print("\n")

            p.train(train_input[i], train_output[i])
            loss_sum += np.square(train_output[i] - p.getOutput())
        train_loss.append(loss_sum / float(len(train_output)))

        # calculate loss on test data
        loss_sum = 0.0
        for i in range(len(test_output)):
            loss_sum += np.square(test_output[i] - p.test(test_input[i]))
        avg_loss = loss_sum / float(len(test_output))
        test_loss.append(avg_loss)

        print(j, avg_loss)

    Chart.makeTwoPlots(train_loss, test_loss)
    Chart.show()


