
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy

training_accuracy = numpy.array([0.99071428571428577, 0.99071428571428577, 0.99285714285714288, 0.97785714285714287, 0.5, 0.48714285714285716, 0.30571428571428572])
testing_accuracy = numpy.array([0.97250000000000003, 0.97250000000000003, 0.97375, 0.96999999999999997, 0.5, 0.49249999999999999, 0.3125])
lambdas = numpy.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

x = lambdas
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(x, training_accuracy, c='black')
ax.scatter(x, training_accuracy, c='black')
ax.plot(x, testing_accuracy, c='blue')
ax.scatter(x, testing_accuracy, c='blue')
ax.set(xlabel = 'lambda', ylabel = 'accuracy')
fig.savefig("q2.3-accuracy.png")


