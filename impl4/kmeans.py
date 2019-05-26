import argparse
import random
from collections import Counter

import numpy
import scipy.cluster.vq as vq

FILENAME = "p4-data.txt"
#FILENAME = "smalldata.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="number of clusters to find")
    parser.add_argument("--iter", type=int, default=100, help="number of iterations of k-means to run")
    parser.add_argument("--seed", type=int, default=42, help="seed the initialization parameters")
    args = parser.parse_args()

    numpy.random.seed(args.seed)

    data = read_data()
    print(data.shape)
    centroid, errors = cluster(data, args.k, args.iter)
    print(centroid)
    print(errors)

def read_data():
    with open(FILENAME, "rb") as f:
        return numpy.loadtxt(f, delimiter=",")

def cluster(data, k, iter):
    errors = []
    for iteration in range(iter+1):
        #centroid, label = vq.kmeans2(data, centroid, iter=1, minit='matrix')

        m = data.shape[0]
        # 1. compute new centroids
        if iteration == 0:
            centroid = numpy.array(random.sample(data, k))
            label = numpy.zeros(m, dtype=numpy.int)
        else:
            # average of each point in the cluster
            centroid = numpy.zeros((k, data.shape[1]))
            for i in range(k):
                assert any(label==i) # XXX delete
                centroid[i] = numpy.mean(data[label==i], axis=0)

        # 2. label each point according to which centroid it is closest to
        for i in range(m):
            norm_squared = numpy.sum(numpy.square(centroid - data[i]), axis=1)
            label[i] = numpy.argmax(norm_squared)

        # 3. Calculate error
        sse = compute_error(data, centroid[label])
        errors.append(sse)
        print(sorted(Counter(label).items()))

    return centroid, errors

def compute_error(actual, predicted):
    d = numpy.subtract(actual, predicted)
    return sum(x.dot(x) for x in d)
    e = numpy.inner(d, d)
    print(e.shape)
    return numpy.sum(e)


if __name__ == '__main__':
    main()
