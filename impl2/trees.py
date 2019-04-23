#!/usr/bin/env python3
import math
from collections import Counter

# choose best test for root of tree
#       pick test on a feature
# recursively choose tests for the descendent nodes
# if all data belongs to the same class, make a leaf node
#       or just stop when the error is small enough

# test ideas:
#  - pick feature that divides roughlyl in half
#  - divide data so that one side is more positive and one side is more negative

class Node:
    def __init__(self, feature):
        self.feature = feature
        self.children = []
    def add(value, node):
        self.children.append((value, node))
    def __str__(self):
        # TODO: prettify
        s = "feature %d (%s)" % (self.feature, self.children)
        return s

class Leaf:
    def __init__(self, decision):
        self.decision = decision
    def __str__(self):
        if decision > 0:
            return "+"
        else:
            return "-"

def divide(S, features, depth):
    """generates a decision tree of depth d
    S is an array of feature vectors.
    S[i][0] should be the class of feature i, either +1 or -1.
    only handles continuous features
    """
    if depth <= 0 or not features:
        # return majority class
        positive = len(S[0] == 1)
        negative = len(S[0] == -1)
        if positive >= negative:
            return Leaf(+1)
        return Leaf(-1)

    best_feature = find_best_feature(S, features)

    tree = Node(best_feature)
    for value in best_feature:
        si = split(S, best_feature, value)
        # <remove current feature from feature list>
        subtree = divide(si, features, depth-1)
        node.add(value, subtree)

    return

def find_best_feature(S, features):
    benefit = {}
    for f in features:
        U = []
        ### need to consider all binary splits of f at v
        values = get_boundary_points(S, f)
        b = entropy(S, f)
        for value in f:
            si = split(S, f, value)
            pi = len(si) / len(S)
            b -= entropy(si, f)*pi

        benefit[f] = b

    best_feature = max(features, key=benefit.__getitem__)
    return best_featureo

def entropy(S, f):
    """determine the information theoretic entropy of some feature in S"""
    p = {}
    c = Counter(x[f] for x in S)
    size = len(S)
    for v, count in c.items():
        p[v] = count / size
    return -sum(p[v]*log2(p[v]) for v in c)

def log2(x):
    return math.log(x, 2)

def get_boundary_points(S, f):
    """find the set of boundary points for a continuous feature"""
    values = sorted(set(x[f] for x in S))
    boundary = []
    for v1, v2 in zip(values, values[1:]):
        m = v1 + (v2-v1)/2
        boundary.append(m)
    return boundary

def find_best_split(S, f):
    """finds the best point to split a continuous feature"""
    values = get_boundary_points(S, f)
    benefit = {}
    for value in values:
        a, b = split(S, f, value)
        p = len(a) / len(S)
        benefit[value] = entropy(S, f) - p*entropy(a, f) - (1-p)*entropy(b, f)
    return max(values, key=benefit.__getitem__)

def split(S, f, value):
    """split S into two sets according to whether feature f < value"""
    a = []
    b = []
    for x in S:
        if x[f] < value:
            a.append(x)
        else:
            b.append(x)
    return a, b


# pruning:
#   build a perfect tree on the training set
#   measure error against validation set
#   divide validtion data based on the tree - pump it to the leaf nodes
#   for each feature node directly above a leaf, compute validtion error change if that node removed
#   remove node that improves validtion error the most

import csv

def main():
    with open("p2-data/knn_test.csv", "r") as f:
        r = csv.reader(f)
        data = [list(map(int_or_float, x)) for x in r]
    features = list(range(1, len(data[0])))
    tree = divide(data, features, 1)
    print(tree)

def int_or_float(s):
    if '.' in s:
        return float(s)
    return int(s)


if __name__ == '__main__':
    main()
