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

    features = list(range(1, len(data[0])))
    tree = trees.divide(data, features, 1)

    print(tree)
    print("info gain", tree.gain(data))
    print("training error", test(tree, data))
    print("testing error", test(tree, testing_data))

def test(tree, data):
    errors = 0
    for x in data:
        if x[0] != tree.predict(x):
            errors += 1
    error_percent = errors / len(data)
    return error_percent

main()
