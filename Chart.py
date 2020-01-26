import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


fig, ax = plt.subplots()
ax.set_ylabel('Square loss')
ax.set_xlabel('Iteration')


class Chart:
    @staticmethod
    def plotTrainingTestLosses(data1, data2):
        ax.plot(data1, 'blue', label='Training set loss', picker=0.01)
        ax.plot(data2, 'red', label='Test set loss', picker=0.01)
        max_value = max(data1[0])
        ax.set_yticks(list(np.arange(0, max_value, 0.01)), minor=True)
        ax.set_xticks(list(range(0, len(data1), 10)), minor=True)
        ax.legend(loc='upper right')

    @staticmethod
    def addToPlot(data, label=''):
        ax.plot(data, label=label, picker=0.01)
        ax.legend()
        plt.yscale("log")

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save(size=(640, 480), dpi=72):
        size = [x/dpi for x in size]
        fig.set_size_inches(size, forward=True)
        fig.savefig('chart.png', dpi=dpi)\

    @staticmethod
    def setAxisLabels(y_label, x_label):
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)