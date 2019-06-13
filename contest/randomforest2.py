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
    parser.add_argument("--noclass", help="whether the test file has a class column or not", action='store_true')
    parser.add_argument("train_filename", help="path to training data")
    parser.add_argument("test_filename", help="path to testing data", default="")
    args = parser.parse_args()

    filename = args.train_filename
    ncols = getncols(filename)
    debug(ncols)
    cols = list(range(1,ncols))
    data = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    target = data[:,0].astype(int)
    data = data[:,1:]
    #print(names[:10])
    debug(data[:1])
    debug(target[:10])

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    debug(rf)
    rf = rf.fit(data, target)

    debug("train:", rf.score(data, target))

    if args.test_filename:
        filename = args.test_filename
    names = numpy.loadtxt(filename, skiprows=1, usecols=[0], dtype=str)
    if args.noclass:
        cols = list(range(1,ncols-1))
        testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
    else:
        testdata = numpy.loadtxt(filename, skiprows=1, usecols=cols)
        testtarget = testdata[:,0].astype(int)
        testdata = testdata[:,1:]

        debug("test:", rf.score(testdata, testtarget))
        debug("neg acc:", rf.score(testdata[testtarget==0], testtarget[testtarget==0]))
        debug("pos acc:", rf.score(testdata[testtarget==1], testtarget[testtarget==1]))

    for name, prob in zip(names, rf.predict_proba(testdata)):
        print(name, prob[1], sep=",")


main()
