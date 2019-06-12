#!/usr/bin/env python3
import csv
import math
import textwrap
from collections import Counter
import numpy
import sklearn.ensemble

def main():
    filename = "resampled.txt"
    cols = list(range(1,105))
    data = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    target = data[:,0].astype(int)
    data = data[:,1:]
    #print(names[:10])
    print(data[:1])
    print(target[:10])

    filename = "feature103_Train.txt"
    names = numpy.loadtxt(filename, skiprows=1, usecols=[0], dtype=str)
    testdata = numpy.loadtxt("feature103_Train.txt", skiprows=1, usecols=cols)
    testtarget = testdata[:,0].astype(int)
    testdata = testdata[:,1:]

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    print(rf)

    rf = rf.fit(data, target)
    pred = rf.predict(data)
    acc = sum(a == b for a, b in zip(pred, target))
    print(acc/float(len(data)))

    pred = rf.predict(testdata)
    acc = sum(a == b for a, b in zip(pred, testtarget))
    print(acc/float(len(testdata)))

    for name, prob in zip(names, rf.predict_proba(testdata)):
        print(name, prob[1], sep=",")


main()
