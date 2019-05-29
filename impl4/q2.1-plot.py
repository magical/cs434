import numpy
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

with open("kmeans.data") as f:
    data = numpy.loadtxt(f, delimiter="\t")

data = data[data[:,0] == 2.0]
data = data[:21] # grab a single run

print(data)
fig, ax = plt.subplots()
ax.set(xlabel="iteration", ylabel="SSE",
    title="SSE after n iterations of k-NN with k=2")
ax.plot(range(21), data[:,-1].reshape(21), marker='o')
fig.savefig("kmeans-1.png")
