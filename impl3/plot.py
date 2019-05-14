#!/usr/bin/env python2
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

#prefix = "q3-d0.1-m0.5-wd0.001"
#datafilename = "q3-model/" + prefix + ".data"

prefix = "q1-lr0.1"
datafilename = "q1-model/" + prefix + ".data"

def read_data(datafilename):
    with open(datafilename) as f:
        lossv = f.readline().split()
        assert lossv[0] == "loss"
        accv = f.readline().split()
        assert accv[0] == "accuracy"
        lossv = [float(x) for x in lossv[1:]]
        accv = [float(x) for x in accv[1:]]
    return lossv, accv

for lr in "0.1", "0.01", "0.001", "0.0001":
    prefix = "q1-lr{}".format(lr)
    datafilename = "q1-model/" + prefix + ".data"

    lossv, accv = read_data(datafilename)
    x = list(range(1,len(lossv)+1))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0.0, 2.5)
    ax2.set_ylim(0.0, 50.0)
    l1, = ax1.plot(x, lossv, c='black', marker='o')
    l2, = ax2.plot(x, accv, c='blue', marker='s')
    ax1.set(xlabel = 'Epoch', ylabel = 'Loss')
    ax2.set(ylabel = 'Accuracy')
    ax1.legend([l1, l2], ["Epoch", "Accuracy"])
    fig.savefig(prefix+".png")
