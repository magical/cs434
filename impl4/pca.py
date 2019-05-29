import sys
import numpy

FILENAME = "p4-data.txt"

def read_data():
    with open(FILENAME, "rb") as f:
        return numpy.loadtxt(f, delimiter=",")

def pca(data, k):
    numpy.set_printoptions(threshold=sys.maxsize)
    # 1. compute covariance matrix
    # easy version:
    #   V = numpy.cov(numpy.transpose(data))
    mu = numpy.mean(data, axis=0)
    V = numpy.zeros((mu.shape[0], mu.shape[0]))
    for x in data:
        t = x - mu
        V += numpy.outer(t, t)
    V /= float(data.shape[0])
    # 2. find eigenvectors
    el, ev = numpy.linalg.eig(V)
    # 3. select top k eigenvectors
    top = numpy.argsort(el)[::-1]
    return numpy.transpose(ev)[top[:k]], el[top[:k]]


