import numpy
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt

with open("kmeans.data") as f:
    data = numpy.loadtxt(f, delimiter="\t")

ks = sorted(list(set(data[:,0])))
best = numpy.zeros((len(ks)))
best = []
for i, k in enumerate(ks):
    sample = data[data[:,0] == k]
    best.append(numpy.min(sample[:,2]))

print(data)
fig, ax = plt.subplots()
ax.set(xlabel="k", ylabel="SSE",
    title="SSE for various k values in k-means, best out of 10 trials")
ax.plot(ks, best, marker='o')
fig.savefig("kmeans-2.png")
