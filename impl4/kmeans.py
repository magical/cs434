import argparse
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
    print(cluster(data, args.k, args.iter)[1])

def read_data():
    with open(FILENAME, "rb") as f:
        return numpy.loadtxt(f, delimiter=",")

def cluster(data, k, iter):
    errors = []
    for i in range(iter):
        if i == 0:
            centroid, label = vq.kmeans2(data, k, iter=1, minit='points')
        else:
            centroid, label = vq.kmeans2(data, centroid, iter=1, minit='matrix')
        sse = compute_error(data, centroid[label])
        errors.append(sse)
    return centroid, errors

def compute_error(actual, predicted):
    d = numpy.subtract(actual, predicted)
    return sum(x.dot(x) for x in d)
    e = numpy.inner(d, d)
    print(e.shape)
    return numpy.sum(e)


if __name__ == '__main__':
    main()
