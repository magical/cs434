#!/usr/bin/env python3
import csv
import math
import textwrap
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
    def __init__(self, S, feature, split_point):
        #self.S = S
        self.feature = feature
        self.split = split_point
        self.entropy = entropy(S)
        self.positive = sum(1 for x in S if x[0] == 1)
        self.negative = len(S) - self.positive
        self.children = []
    def gain(self, S):
        a, b = split(S, self.feature, self.split)
        return calc_information_gain(S, a, b)
    def add(self, value, node):
        self.children.append((value, node))
    def predict(self, x):
        if x[self.feature] < self.split:
            return self.children[0][1].predict(x)
        else:
            return self.children[1][1].predict(x)
    def __str__(self):
        labels = ["<", "≥"]
        s = "feature %d ratio=%d:%d entropy=%f (\n%s)" % (
                self.feature,
                self.positive,
                self.negative,
                self.entropy,
                ",\n".join(textwrap.indent("feature %d %s %f -> %s" % (self.feature, l, self.split, node), "  ")
                        for (v, node), l in zip(self.children, labels)))
        return s

class Leaf:
    def __init__(self, S):
        self.positive = sum(1 for x in S if x[0] == 1)
        self.negative = len(S) - self.positive
        if self.positive >= self.negative:
            self.decision = +1
        else:
            self.decision = -1
    def predict(self, x):
        return self.decision
    def __str__(self):
        if self.decision > 0:
            return "leaf +1 (ratio=%d:%d)" % (self.positive, self.negative)
        else:
            return "leaf -1 (ratio=%d:%d)" % (self.positive, self.negative)

def divide(S, features, depth):
    """generates a decision tree of depth d
    S is an array of feature vectors.
    S[i][0] should be the class of feature i, either +1 or -1.
    only handles continuous features
    """
    if depth <= 0 or not features:
        # return majority class
        return Leaf(S)
    if len(set(x[0] for x in S)) == 1:
        return Leaf(S)

    best_feature, best_value = find_best_feature(S, features)

    tree = Node(S, best_feature, best_value)
    a, b = split(S, best_feature, best_value)
    for si in a, b:
        # TODO: remove current feature from feature list?
        subtree = divide(si, features, depth-1)
        tree.add(best_value, subtree)

    return tree

def find_best_feature(S, features):
    benefit = {}
    values = {}
    for f in features:
        value, gain = find_best_split(S, f)
        values[f] = value
        benefit[f] = gain

    best_feature = max(features, key=benefit.__getitem__)
    return best_feature, values[best_feature]

def entropy(S):
    """determine the information theoretic entropy of the labels in S"""
    counter = Counter(x[0] for x in S)
    size = float(len(S))
    ps = [count/size for (v, count) in counter.items()]
    return -sum(p*log2(p) for p in ps)

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
        benefit[value] = calc_information_gain(S, a, b)
    best_value = max(values, key=benefit.__getitem__)
    return best_value, benefit[best_value]

def calc_information_gain(S, a, b):
    """calculate the information gain from splitting S into sets a and b"""
    p = len(a) / len(S)
    H = entropy(S) - p*entropy(a) - (1-p)*entropy(b)
    return H

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

def read_data(filename):
    with open(filename, "r") as f:
        r = csv.reader(f)
        data = [list(map(int_or_float, x)) for x in r]
    assert all(x[0] in (+1, -1) for x in data)
    return data

# pruning:
#   build a perfect tree on the training set
#   measure error against validation set
#   divide validtion data based on the tree - pump it to the leaf nodes
#   for each feature node directly above a leaf, compute validtion error change if that node removed
#   remove node that improves validtion error the most
#
# we aren't doing pruning though


def main():
    with open("p2-data/knn_test.csv", "r") as f:
        r = csv.reader(f)
        data = [list(map(int_or_float, x)) for x in r]
    features = list(range(1, len(data[0])))
    tree = divide(data, features, 3)
    print(tree)

def int_or_float(s):
    if '.' in s:
        return float(s)
    return int(s)


if __name__ == '__main__':
    main()
