#!/usr/bin/env python3

import sys
import numpy

def main():
    training_file = sys.argv[1]
    testing_file = sys.argv[2]

    training_data, training_target = load_data(training_file)
    testing_data, testing_target = load_data(testing_file)

    w = train(training_data, training_target)
    print("training ASE = ", evaluate(w, training_data, training_target))
    print("testing ASE  = ", evaluate(w, testing_data, testing_target))

def load_data(filename):
    """load space-separated floating-point data"""
    data = []
    target = []
    with open(filename) as f:
        for line in f:
            row = line.split()
            data.append([float(x) for x in row[:-1]])
            target.append(float(row[-1]))
    return data, target

def train(features, target):
    # Linear regression
    Y = numpy.array(target)
    X = numpy.array(features, dtype=numpy.float32)
    # DON't add 1s column
    #X = numpy.concatenate((numpy.ones((X.shape[0], 1)), X), axis=1)
    X = numpy.matrix(X)
    #print(X)

    w = ((X.T * X).I * X.T).dot(Y)
    w = numpy.ravel(w)
    print(w)
    return w

def evaluate(w, features, target):
    X = numpy.array(features)
    Y = numpy.array(target)
    # DON't add 1s column
    #X = numpy.concatenate((numpy.ones((X.shape[0], 1)), X), axis=1)
    prediction = X.dot(w)

    loss = ase(prediction - Y)
    return loss


def ase(x):
    return numpy.sum(numpy.square(x)) / len(x)

main()
