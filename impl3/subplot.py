#!/usr/bin/env python2
import argparse
import math
import sys
import os

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

def read_data(datafilename):
    with open(datafilename) as f:
        lossv = f.readline().split()
        assert lossv[0] == "loss"
        accv = f.readline().split()
        assert accv[0] == "accuracy"
        lossv = [float(x) for x in lossv[1:]]
        accv = [float(x) for x in accv[1:]]
    return lossv, accv

def make_plot(datafilename, ax1, ax2):
    lossv, accv = read_data(datafilename)
    x = list(range(1,len(lossv)+1))

    ax1.set_ylim(0.0, 2.5)
    ax2.set_ylim(0.0, 50.0)
    l1, = ax1.plot(x, lossv, c='black', marker='o')
    l2, = ax2.plot(x, accv, c='blue', marker='s')
    #ax1.set(xlabel = 'Epoch', ylabel = 'Loss')
    #ax2.set(ylabel = 'Accuracy')
    return l1, l2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    filenames = args.filenames
    n = len(filenames)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    base = "figure"
    for i, datafilename in enumerate(filenames):
        r = i // cols
        c = i % cols
        ax1 = axes[r,c]
        ax2 = ax1.twinx()
        if r < rows-1:
            ax1.tick_params(axis='x', bottom=False, labelbottom=False)
        if c < cols-1:
            ax2.tick_params(axis='y', right=False, labelright=False)
        if c > 0:
            ax1.tick_params(axis='y', left=False, labelleft=False)
        if c == cols-1:
            ax2.set(ylabel="Accuracy")
        if c == 0:
            ax1.set(ylabel="Loss")
        if r == rows-1:
            ax1.set(xlabel="Epochs")
        base, _ = os.path.splitext(os.path.basename(datafilename))
        lines = make_plot(datafilename, ax1, ax2)
    
    outfilename = base+".png"
    if args.output:
        outfilename = args.output
    #fig.set(xlabel='Epoch', ylabel="Accuracy")
    fig.legend(lines, ["Loss", "Accuracy"], loc='upper center')
    fig.savefig(outfilename)

    #for lr in "0.1", "0.01", "0.001", "0.0001":
    #    prefix = "q1-lr{}".format(lr)
    #    datafilename = "q1-model/" + prefix + ".data"
    #    make_plot(datafilename, prefix+".png")

main()
