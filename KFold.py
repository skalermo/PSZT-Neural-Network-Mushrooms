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

        # Chunk size
        chunk_size = int(len(data) / self.k)

        # Shuffle arrays
        if self.shuffle:
            self.rand.shuffle(idx)

        # Split into chunks (train = data - test), len(test) = chunk_size
        for i in range(0, len(idx), chunk_size):
            test = idx[i:i + chunk_size]
            train = idx[0:i] + idx[i + chunk_size:]
            yield train, test


