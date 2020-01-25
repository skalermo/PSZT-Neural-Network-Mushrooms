from time import time
from random import Random


class KFold:
    def __init__(self, k, shuffle=True, seed=time()):
        self.k = k
        self.shuffle = shuffle

        # If data is going to be shuffled create Random object
        if shuffle:
            self.rand = Random()
            self.rand.seed(seed)

    # Split data into k-length chunks
    def split(self, data):
        # Create index list
        idx = list(range(len(data)))

        # Shuffle arrays
        if self.shuffle:
            self.rand.shuffle(idx)

        # Split into chunks (train = data - test), len(test) = k
        for i in range(0, len(idx), self.k):
            test = idx[i:i + self.k]
            train = idx[0:i] + idx[i + self.k:]
            yield train, test


