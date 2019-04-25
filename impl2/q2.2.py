#!/usr/bin/env python3

import sys
import trees

def getarg(n, default):
    if len(sys.argv) > n:
        return sys.argv[n]
    return default

def main():
    data = trees.read_data(getarg(1, "p2-data/knn_train.csv"))
    testing_data = trees.read_data(getarg(2, "p2-data/knn_test.csv"))
    d = getarg(3, None)

    if d is not None:
        ds = int(d)
    else:
        ds = range(1,6+1)

    features = list(range(1, len(data[0])))
    for d in ds:
        tree = trees.divide(data, features, d)
        print(d, test(tree, data), test(tree, testing_data), sep="\t")
        print(tree, file=sys.stderr)

def test(tree, data):
    errors = 0
    for x in data:
        if x[0] != tree.predict(x):
            errors += 1
    error_percent = errors / len(data)
    return error_percent

main()
