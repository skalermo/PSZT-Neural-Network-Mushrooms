from argparse import ArgumentParser
from time import time


def parseArgs():
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

    return vars(parser.parse_args())