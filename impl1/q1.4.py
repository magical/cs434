#!/usr/bin/env python3

import os
import sys
import numpy


training_accuracy = []
testing_accuracy = []

def main():
    training_file = sys.argv[1]
    testing_file = sys.argv[2]

    real_training_data, training_target = load_data(training_file)
    real_testing_data, testing_target = load_data(testing_file)

    ds = []
    for d in 2, 4, 6, 8, 10:
        training_data = add_dummy_columns(real_training_data, d)
        testing_data = add_dummy_columns(real_testing_data, d)
        w = train(training_data, training_target)
        print("d = ", d)
        print("training ASE = ", evaluate(w, training_data, training_target))
        print("testing ASE  = ", evaluate(w, testing_data, testing_target))
        print()
        training_accuracy.append(evaluate(w, training_data, training_target))
        testing_accuracy.append(evaluate(w, testing_data, testing_target))
        ds.append(d)

    with open("q1.4-plot.py", "w") as f:
        f.write(graph_py % (training_accuracy, testing_accuracy, ds))
    os.system("python2 q1.4-plot.py")


def load_data(filename):
    """load space-separated floating-point data"""
    data = []
    target = []
    with open(filename) as f:
        for line in f:
            row = line.split()
            data.append([float(x) for x in row[:-1]])
            target.append(float(row[-1]))
    return numpy.array(data, dtype=numpy.float32), numpy.array(target, dtype=numpy.float32)

def add_dummy_columns(data, n):
    dummy_data = numpy.random.standard_normal((data.shape[0], n))
    return numpy.concatenate((data, dummy_data), axis=1)


def train(features, target):
    # Linear regression
    Y = numpy.array(target)
    X = numpy.array(features, dtype=numpy.float32)
    X = numpy.matrix(X)

    w = ((X.T * X).I * X.T).dot(Y)
    w = numpy.ravel(w)
    print(w)
    return w

def evaluate(w, features, target):
    X = numpy.array(features)
    Y = numpy.array(target)
    prediction = X.dot(w)

    loss = ase(prediction - Y)
    return loss


def ase(x):
    return numpy.sum(numpy.square(x)) / len(x)

graph_py = r"""
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

training_accuracy = numpy.array(%r)
testing_accuracy = numpy.array(%r)
x = numpy.array(%r)

fig, ax = plt.subplots()
ax.plot(x, training_accuracy, c='black')
ax.scatter(x, training_accuracy, c='black')
ax.plot(x, testing_accuracy, c='blue')
ax.scatter(x, testing_accuracy, c='blue')
ax.set(xlabel = 'lambda', ylabel = 'accuracy')
fig.savefig("q1.4-accuracy.png")
"""

main()
