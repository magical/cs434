import numpy

FILENAME = "p4-data.txt"

def read_data():
    with open(FILENAME, "rb") as f:
        return numpy.loadtxt(f, delimiter=",")

def pca(data, k):
    # 1. compute covariance matrix
    V = numpy.cov(numpy.transpose(data))
    # 2. find eigenvectors
    el, ev = numpy.linalg.eig(V)
    # 3. select top k eigenvectors
    top = numpy.argsort(el)[::-1]
    return numpy.transpose(ev)[top[:k]], el[top[:k]]


