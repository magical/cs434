#!/usr/bin/env python3

import os
import sys
import csv
import numpy

eta = 1.0
lambda_ = 1.0

training_accuracy = []
testing_accuracy = []

def main():
    global eta
    global lambda_
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    if len(sys.argv) > 3:
        lambdas = [float(x) for x in sys.argv[3].split(",")]
    else:
        lambdas = [0.0]

    training_data, training_target = load_data(training_file)
    testing_data, testing_target = load_data(testing_file)

    print("eta = ", eta)
    for lambda_ in lambdas:
        print("lambda = ", lambda_)
        w = descend_batch(training_data, training_target, testing_data, testing_target, lambda_)
        tr_acc = evaluate(w, training_data, training_target)
        te_acc = evaluate(w, testing_data, testing_target)
        print("w = ", w)
        print("training accuracy = ", tr_acc)
        print("testing accuracy  = ", te_acc)
        training_accuracy.append(tr_acc)
        testing_accuracy.append(te_acc)

    with open("q2.3-plot.py", "w") as f:
        f.write(graph_py % (training_accuracy, testing_accuracy, lambdas))
    os.system("python2 q2.3-plot.py")

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
    data /= numpy.max(data) # normalize features
    return data, target

def descend_batch(features, target, testing_data, testing_target, lambda_):
    X = features
    Y = target
    # batch gradient descent
    #print(X.shape)
    batches = 50
    w = numpy.zeros(X.shape[1])
    for i in range(batches):
        grad = numpy.zeros(X.shape[1])
        for x, y in zip(X, Y):
            yhat = sigma(w.dot(x))
            #print("grad +=", (yhat - y) * x)
            grad += (yhat - y) * x
        #print(grad[:10])
        reg = lambda_ * w
        grad += reg
        w = w - eta*grad
        #print(w[:10])

        # TODO: for each iteration, plot the accuracy
        #print(w)
        #print("eval", evaluate(w, X, Y))
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

graph_py = r"""
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

training_accuracy = numpy.array(%r)
testing_accuracy = numpy.array(%r)
lambdas = numpy.array(%r)

x = lambdas
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(x, training_accuracy, c='black')
ax.scatter(x, training_accuracy, c='black')
ax.plot(x, testing_accuracy, c='blue')
ax.scatter(x, testing_accuracy, c='blue')
ax.set(xlabel = 'lambda', ylabel = 'accuracy')
fig.savefig("q2.3-accuracy.png")


"""

main()
