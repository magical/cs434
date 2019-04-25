#!/usr/bin/python2
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

with open("q2.data") as f:
    data = numpy.loadtxt(f)

labels = ['Training error', 'Testing error']
colors = ["black", "blue"]
markers = list("^v")


x = data[:,0]

fig, ax = plt.subplots()
lines = []
for i in range(1, data.shape[1]):
    l, = ax.plot(x, data[:,i], c=colors[i-1], marker=markers[i-1])
    lines.append(l)
ax.set(xlabel = 'depth', ylabel = 'error percentage')
ax.legend(lines, labels, loc='best', fancybox=True, shadow=True)
fig.savefig("q2-errors.png")
