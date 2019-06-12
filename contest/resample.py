#!/usr/bin/env python3
import random
import numpy

def main():
    random.seed(4)
    target = numpy.loadtxt("feature103_Train.txt", skiprows=1, usecols=[1], dtype=int)
    with open("feature103_Train.txt") as f:
        print(next(f), end="")
        lines = list(f)
    # sample 13,000 features from each class, with replacement
    neg = [i for i in range(len(target)) if target[i] == 0]
    pos = [i for i in range(len(target)) if target[i] == 1]
    for i in range(13000):
        n = random.choice(neg)
        print(lines[n], end="")
        p = random.choice(pos)
        print(lines[p], end="")

main()
