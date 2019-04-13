#!/usr/bin/env python3

import sys
import csv
import numpy

eta = 0.01

def main():
    global eta
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    if len(sys.argv) > 3:
        eta = float(sys.argv[3])

    training_data, training_target = load_data(training_file)
    testing_data, testing_target = load_data(testing_file)

    print("eta = ", eta)
    w = train(training_data, training_target)
    print(w)
    print("training ASE = ", evaluate(w, training_data, training_target))
    print("testing ASE  = ", evaluate(w, testing_data, testing_target))

def load_data(filename):
    """load space-separated floating-point data"""
    data = []
    target = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row[:-1]])
            target.append(float(row[-1]))
    data = numpy.array(data, dtype=numpy.float32)
    target = numpy.array(target, dtype=numpy.float32)
    return data, target

def train(features, target):
    return descend_batch(features, target)

def descend_simple():
    # Simple gradient descent
    while True:
        for i in range(n):
            y = Y[i]
            x = X[i]
            yhat = sigma(w.dot(x))
            w = w - eta*(yhat - y)*x

def descend_batch(features, target):
    X = features
    Y = target
    # batch gradient descent
    #print(X.shape)
    batches = 200
    w = numpy.zeros(X.shape[1])
    for i in range(batches):
        grad = numpy.zeros(X.shape[1])
        for x, y in zip(X, Y):
            yhat = sigma(w.dot(x))
            #print("grad +=", (yhat - y) * x)
            grad += (yhat - y) * x
        #print(grad[:10])
        w = w - eta*grad
        print(w[:10])

        # TODO: for each iteration, plot the accuracy
        #print(w)
        print("eval", evaluate(w, X, Y))
    return w

def sigma(a):
    return 1. / (1. + numpy.exp(-a))

def evaluate(w, X, Y):
    # accuracy:
    p = sigma(X.dot(w))
    #return ase(p-Y)
    yes = ~numpy.logical_xor(p>.5, Y)
    #print(p[:10])
    #print(Y[:10])
    #print(yes[:10])
    return sum(yes) / len(yes)


def logloss(w, X, Y):
    p = sigma(X.dot(w))
    return - numpy.sum(Y*numpy.log(p) + (1-Y)*numpy.log(1-p))

def ase(x):
    return numpy.sum(numpy.square(x)) / len(x)

main()
