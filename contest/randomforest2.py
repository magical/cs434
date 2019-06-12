#!/usr/bin/env python3
import csv
import math
import textwrap
from collections import Counter
import numpy
import sklearn.ensemble

def main():
    filename = "small.txt"
    cols = list(range(1,105))
    data = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    target = data[:,0].astype(int)
    data = data[:,1:]
    #print(names[:10])
    print(data[:1])
    print(target[:10])

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    print(rf)
    rf = rf.fit(data, target)

    pred = rf.predict(data)
    acc = sum(a == b for a, b in zip(pred, target))
    print(acc/float(len(data)))

    filename = "feature103_Train.txt"
    names = numpy.loadtxt(filename, skiprows=1, usecols=[0], dtype=str)
    testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    testtarget = testdata[:,0].astype(int)
    testdata = testdata[:,1:]

    pred = rf.predict(testdata)
    acc = sum(a == b for a, b in zip(pred, testtarget))
    print(acc/float(len(testdata)))
    print(rf.score(testdata, testtarget))
    print("neg acc:", rf.score(testdata[testtarget==0], testtarget[testtarget==0]))
    print("pos acc:", rf.score(testdata[testtarget==1], testtarget[testtarget==1]))

    for name, prob in zip(names, rf.predict_proba(testdata)):
        print(name, prob[1], sep=",")


main()
