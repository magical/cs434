
import csv
import sys

import numpy

class KNN:
    def __init__(self, xs, y, k):
        self.xs = xs
        self.y = y
        self.k = k

    def predict(self, data):
        ps = []
        for point in data:
            dists = []
            for x in self.xs:
                delta = numpy.subtract(point, x)
                d = numpy.dot(delta, delta)
                dists.append(d)
            dists = numpy.argpartition(dists, self.k)
            n = sum(self.y[i] for i in dists[:self.k]) # assume y is +1 or -1
            prediction = +1 if n > 0 else -1 # majority class
            ps.append(prediction)
        return ps

    def test(self, data, target):
        prediction = self.predict(data)
        error = numpy.not_equal(prediction, target)
        return sum(error)


    def validate(self):
        """perform leave-one-out cross validation.
        returns the number of misclassified points"""

        errors = 0
        # for each point in the data set
        for i in range(len(self.xs)):
            point = self.xs[i]
            dists = []
            # try to predict that one point, ignoring it for the purposes of finding neighbors
            for j in range(len(self.xs)):
                if i == j:
                    dists.append(float("inf"))
                    continue
                delta = numpy.subtract(point, self.xs[j])
                d = numpy.dot(delta, delta)
                dists.append(d)
            dists = numpy.argpartition(dists, self.k)
            n = sum(self.y[i] for i in dists[:self.k]) # assume y is +1 or -1
            prediction = +1 if n > 0 else -1 # majority class
            if prediction != self.y[i]:
                errors += 1

        return errors


def int_or_float(s):
    if '.' in s:
        return float(s)
    return int(s)

def read_csv(filename):
    with open(filename, "r") as f:
        r = csv.reader(f)
        data = [list(map(int_or_float, x)) for x in r]
    target = [row.pop(0) for row in data]
    assert set(target) == {-1, +1}

    # normalize data to range [0,1]
    data = numpy.array(data)
    data -= numpy.min(data, axis=0)
    data /= numpy.max(data, axis=0)
    data = list(data)

    return data, target

def getarg(n, default):
    if len(sys.argv) > n:
        return sys.argv[n]
    return default

def main():
    verbose = False
    training_filename = getarg(1, "p2-data/knn_train.csv")
    testing_filename = getarg(2, "p2-data/knn_test.csv")

    data, target = read_csv(training_filename)
    test_data, test_target = read_csv(testing_filename)

    ks = range(1, 51+1, 2)
    if len(sys.argv) > 3:
        ks = [int(sys.argv[3])]

    for k in ks:
        knn = KNN(data, target, k)
        if verbose:
            print("k", k)
            print("validation", knn.validate())
            print("training error", knn.test(data, target))
            print("testing error", knn.test(test_data, test_target))
            print()
        else:
            print(k, knn.validate(), knn.test(data, target), knn.test(test_data, test_target), sep="\t")

if __name__ == '__main__':
    main()
