import numpy as np
import pandas as pd
from Perceptron import Perceptron
from Chart import Chart


if __name__ == '__main__':
    # Read file
    data = pd.read_csv('dataset.csv')
    test_data = pd.read_csv('controlset.csv')

    # Get output
    Y = data['class']
    testY = test_data['class']

    # Get input
    X = data.drop('class', axis=1)
    testX = test_data.drop('class', axis=1)
    # X['ones'] = np.ones(X.shape[0])

    # Create NN
    input_size = X.shape[1]
    output_size = Y.shape[1] if len(Y.shape) > 1 else 1
    nn = Perceptron(input_size, output_size, hidden_neurons=2)

    # Convert to np arrays
    x = X.to_numpy(float)
    y = Y.to_numpy(float)
    testx = testX.to_numpy(float)
    testy = testY.to_numpy(float)

    # Add one more dimension to input array
    x.shape += (1,)
    testx += (1,)

    train_loss = []
    test_loss = []

    for j in range(100):
        loss_sum = 0.0
        for i in range(len(y)):
            # if i == len(y)-1:
            #     print("for iteration # " + str(i) + "\n")
            #     print("Actual Output: \n" + str(y[i]))
            #     print("Predicted Output: \n" + str(nn.output))
            #     print("Loss: \n" + str((np.square(y[i] - nn.getOutput()))))  # mean sum squared loss
            #     print("\n")

            nn.train(x[i], y[i])
            loss_sum += np.square(y[i] - nn.getOutput())
        train_loss.append(loss_sum / float(len(y)))

        # calculate loss on test data
        loss_sum = 0.0
        for j in range(len(testy)):
            loss_sum += np.square(y[j] - nn.test(x[j]))
        test_loss.append(loss_sum / float(len(testy)))

    Chart.makeTwoPlots(train_loss, test_loss)
    Chart.show()


