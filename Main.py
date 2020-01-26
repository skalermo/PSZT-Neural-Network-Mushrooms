import numpy as np
import pandas as pd
from Perceptron import Perceptron
from KFold import KFold
from argparse import ArgumentParser
from time import time

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

def calcLoss(input, output):
    loss = 0.0
    for i in range(len(input)):
        p.test(input[i])
        loss += np.square(output[i] - p.getOutput())

    return loss / len(output)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Main.py', description='Neural network')
    parser.add_argument('-n', type=int, default=4, metavar='', help='Number of neurons in hidden layer')
    parser.add_argument('-i', type=int, default=100, metavar='', help='Number of iterations')
    parser.add_argument('-l', type=float, default=0.1, metavar='', help='Learning rate')
    parser.add_argument('-r', type=float, default=5.0, metavar='', help='Ratio Training to Validation 1:R')
    parser.add_argument('-k', type=float, default=5.0, metavar='', help='Ratio for K-fold cross validation 1:K')
    parser.add_argument('-s', type=float, default=time(), metavar='', help='Random Seed')
    parser.add_argument('-f', type=str, metavar='', help='Load file')
    parser.add_argument('-c', action='store_true', default=False, help='Output a chart')
    parser.add_argument('-v', action='store_true', default=False, help='Verbose output')

    args = vars(parser.parse_args())

    input, output = readData('dataset.csv')

    # Create Perceptron
    p = Perceptron(input.shape[1], output.shape[1], hidden_neurons=args['n'], learning_rate=args['l'])

    # Import Chart
    if args['c']:
        chart_loss = []
        from Chart import Chart

    # Split data to validation and training sets
    kfold = KFold(args['r'], True, args['s'])

    training_idx, validation_idx = next(kfold.split(input))

    in_training = input[training_idx]
    out_training = output[training_idx]

    in_validation = input[validation_idx]
    out_validation = output[validation_idx]

    # K-Fold cross validation
    test_loss = []
    k = 1

    kfold = KFold(args['k'], True, args['s'])
    for train_idx, test_idx in kfold.split(in_training):
        train_input = in_training[train_idx]
        train_output = out_training[train_idx]
        test_input = in_training[test_idx]
        test_output = out_training[test_idx]

        # Train perceptron
        for iteration in range(args['i']):
            for i in range(len(train_output)):
                p.train(train_input[i], train_output[i])

        # Calculate loss
        loss_sum = 0.0
        for i in range(len(test_input)):
            loss_sum += np.square(test_output[i] - p.test(test_input[i]))

        test_loss.append(loss_sum / len(train_output))

        # For charting purposes
        if args['c']:
            chart_loss.append(calcLoss(in_validation, out_validation))

        if args['v']:
            print("for k # " + str(k))
            print("Loss: " + str(test_loss[-1]))
            print()

        k += 1

    # Test loss
    test_loss = np.average(test_loss)

    # Final perceptron training
    for iteration in range(args['i']):
        for i in range(len(in_training)):
            p.train(in_training[i], out_training[i])

    # Validate
    predicted = 0
    validation_loss = 0.0
    for i in range(len(in_validation)):
        p.test(in_validation[i])
        validation_loss += np.square(out_validation[i] - p.getOutput())

        if args['v']:
            if out_validation[i] != np.round(p.getOutput()):
                print("Actual:{} Predicted:{}".format(out_validation[i], np.round(p.getOutput())))
            predicted += int(out_validation[i] == np.round(p.getOutput()))

    validation_loss = validation_loss / len(out_validation)

    print("Validation Loss: " + str(validation_loss))
    print("Predicted: {}/{} = {}%".format(predicted, len(out_validation), (predicted*100)/len(out_validation)))

    # Show chart
    if args['c']:
        Chart.setAxisLabels('Loss', 'K-Fold iteration')
        Chart.addToPlot(chart_loss, "Validation Loss")
        Chart.show()


