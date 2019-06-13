#!/usr/bin/env python3
import sys
import argparse
import numpy
import sklearn.ensemble

def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def getncols(filename):
    with open(filename) as f:
        _ = f.readline()
        l = f.readline()
        return l.rstrip().count('\t') + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_filename", help="path to training data")
    parser.add_argument("test_filename", help="path to testing data", nargs="?")
    args = parser.parse_args()

    filename = args.train_filename
    ncols = getncols(filename)
    cols = list(range(1,ncols))
    data = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    target = data[:,0].astype(int)
    data = data[:,1:]
    #debug(names[:10])
    debug(data[:1])
    debug(target[:10])

    weight = numpy.zeros_like(target, dtype=float)
    neg = float(sum(target==0))
    pos = float(sum(target==1))
    debug(neg, pos)
    weight[target==0] = (neg+pos)/2 / neg
    weight[target==1] = (neg+pos)/2 / pos
    # the weights should still sum up to len(data)
    debug(sum(weight), len(data))

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    debug(rf)
    rf = rf.fit(data, target, sample_weight=weight)

    debug("train:", rf.score(data, target))

    if args.test_filename:
        filename = args.test_filename
    names = numpy.loadtxt(filename, skiprows=1, usecols=[0], dtype=str)
    if args.noclass:
        cols = range(1,ncols-1)
        testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    else:
        testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
        testtarget = testdata[:,0].astype(int)
        testdata = testdata[:,1:]

        debug(rf.score(testdata, testtarget))
        debug("neg acc:", rf.score(testdata[testtarget==0], testtarget[testtarget==0]))
        debug("pos acc:", rf.score(testdata[testtarget==1], testtarget[testtarget==1]))

    for name, prob in zip(names, rf.predict_proba(testdata)):
        print(name, prob[1], sep=",")


main()
