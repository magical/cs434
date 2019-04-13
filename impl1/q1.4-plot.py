
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

training_accuracy = numpy.array([24.420940962668983, 24.325297899713494, 24.267644500344957, 24.03860263265279, 23.919323174248756])
testing_accuracy = numpy.array([24.343826429135355, 24.492202582337143, 24.792762586721743, 25.458460320994821, 23.760914062931718])
x = numpy.array([2, 4, 6, 8, 10])

fig, ax = plt.subplots()
ax.plot(x, training_accuracy, c='black')
ax.scatter(x, training_accuracy, c='black')
ax.plot(x, testing_accuracy, c='blue')
ax.scatter(x, testing_accuracy, c='blue')
ax.set(xlabel = 'number of random features', ylabel = 'average squared error')
fig.savefig("q1.4-errors.png")
