#!/usr/bin/env python2
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

with open("q1.data") as f:
    data = numpy.loadtxt(f)

x = data[:,0]
validation_error = data[:,1]
training_error = data[:,2]
testing_error = data[:,3]

fig, ax = plt.subplots()
l1, = ax.plot(x, validation_error, c='orange', marker='o')
l2, = ax.plot(x, training_error, c='black', marker='^')
l3, = ax.plot(x, testing_error, c='blue', marker='v')
ax.set(xlabel = 'k', ylabel = 'number of errors')
ax.legend([l1, l2, l3], ['Validation error', 'Training error', 'Testing error'], loc='lower right', fancybox=True, shadow=True)
fig.savefig("q1-errors.png")
