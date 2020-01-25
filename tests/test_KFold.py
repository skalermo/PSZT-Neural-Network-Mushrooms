import unittest

from KFold import KFold
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_chunks(self):
        kfold = KFold(3, True)

        data = np.array([
            ['a', 'b', 'c', 'd', 'e'], ['f', 'g', 'h', 'i', 'j'],
            ['z', 'x', 'c', 'v', 'b'], ['n', 'm', ',', '.', '/'],
            [1, 2, 3, 4, 5], [6, 7, 8, 9, 0]
        ])

        k = 0
        for train, test in kfold.split(data):
            self.assertEqual(len(data[train]), 4, 'Train Chunk incorrect length')
            self.assertEqual(len(data[test]), 2, 'Test Chunk incorrect length')
            k += 1

        self.assertEqual(k, 3, 'Incorrect chunks count')


if __name__ == '__main__':
    unittest.main()
