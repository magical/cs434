
import numpy
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

import pca

data = pca.read_data()
ev, el = pca.pca(data, 10)

plt.gray()
fig = plt.figure()
for i, vec in enumerate(ev):
    x = vec.reshape((7*2*2, 7*2*2))
    x = x / numpy.max(x)
    x = x.astype(numpy.float32)

    fig.add_subplot(3,4,i)
    img = plt.imshow(x)

fig.savefig("pca_2.png")

