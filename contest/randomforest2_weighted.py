
import csv
import math
import textwrap
from collections import Counter
import numpy
import sklearn.ensemble

def main():
    filename = "feature103_Train.txt"
    cols = list(range(1,105))
    data = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    target = data[:,0].astype(int)
    data = data[:,1:]
    #print(names[:10])
    print(data[:1])
    print(target[:10])

    weight = numpy.zeros_like(target, dtype=float)
    neg = float(sum(target==0))
    pos = float(sum(target==1))
    print(neg, pos)
    weight[target==0] = (neg+pos)/2 / neg
    weight[target==1] = (neg+pos)/2 / pos
    # the weights should still sum up to len(data)
    print(sum(weight), len(data))

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    print(rf)
    rf = rf.fit(data, target, sample_weight=weight)

    print(rf.score(data, target))

    filename = "small.txt"
    names = numpy.loadtxt(filename, skiprows=1, usecols=[0], dtype=str)
    testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    testtarget = testdata[:,0].astype(int)
    testdata = testdata[:,1:]

    print(rf.score(testdata, testtarget))
    print("neg acc:", rf.score(testdata[testtarget==0], testtarget[testtarget==0]))
    print("pos acc:", rf.score(testdata[testtarget==1], testtarget[testtarget==1]))

    for name, prob in zip(names, rf.predict_proba(testdata)):
        print(name, prob[1], sep=",")


main()
