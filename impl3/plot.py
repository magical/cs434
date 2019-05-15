#!/usr/bin/env python2
import sys
import os
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

def read_data(datafilename):
    with open(datafilename) as f:
        lossv = f.readline().split()
        assert lossv[0] == "loss"
        accv = f.readline().split()
        assert accv[0] == "accuracy"
        lossv = [float(x) for x in lossv[1:]]
        accv = [float(x) for x in accv[1:]]
    return lossv, accv

def make_plot(datafilename, outfilename):
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
    fig.savefig(outfilename)

def main():
    for datafilename in sys.argv[1:]:
        base, _ = os.path.splitext(os.path.basename(datafilename))
        outfilename = base+".png"
        make_plot(datafilename, outfilename)

    #for lr in "0.1", "0.01", "0.001", "0.0001":
    #    prefix = "q1-lr{}".format(lr)
    #    datafilename = "q1-model/" + prefix + ".data"
    #    make_plot(datafilename, prefix+".png")

main()
