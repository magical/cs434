
import numpy
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

import pca

data = pca.read_data()
ev, el = pca.pca(data, 10)


plt.gray()
fig = plt.figure()

DIM = (28, 28)
fig.add_subplot(3,4,1)
mean = numpy.mean(data, axis=0)
plt.imshow(mean.reshape(DIM) / numpy.max(mean))

for i, vec in enumerate(ev):
    x = vec.reshape(DIM)
    x = x / numpy.max(x)
    x = x.astype(numpy.float32)

    fig.add_subplot(3,4,i+2)
    img = plt.imshow(x)

fig.savefig("pca_2.png")

