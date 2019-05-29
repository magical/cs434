## project data onto the top 10 principle component vectors

import numpy
import pca

data = pca.read_data()
ev, el = pca.pca(data, 10)

projection = numpy.transpose(numpy.dot(ev, numpy.transpose(data)))

best = [0]*10
# find largest in each dimension
for dim in range(10):
    best[dim] = numpy.argmax(projection[:,dim])

# plot the best images
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

DIM = (28, 28)

plt.gray()
fig = plt.figure()

for i in range(len(best)):
    x = data[best[i]].reshape(DIM)
    x /= numpy.max(x)
    fig.add_subplot(3,4,i+2)
    img = plt.imshow(x)

fig.savefig("pca_3.png")

