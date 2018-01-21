# Implementation of gradient descent algorithm on a random data set

# external libraries
import numpy as np
import matplotlib.pyplot as plt


# h(x) = theta0 + theta1*x
def hypothesis(theta0, theta1, x):
    return theta0 + theta1*x


# calculate new values for theta0, theta1 as per gradient descent algorithm
def gradient_step(data, learning_rate, current_theta0, current_theta1):

    # calculate descent for theta0
    d_theta = 0
    for i in range(len(data)):
        d_theta += (hypothesis(current_theta0, current_theta0, data[i][0]) - data[i][1])

    new_theta0 = current_theta0 - (learning_rate * d_theta * 1/float(len(data)))

    # calculate descent for theta1
    d_theta = 0
    for i in range(len(data)):
        d_theta += ((hypothesis(current_theta0, current_theta1, data[i][0]) - data[i][1]) * data[i][0])

    new_theta1 = current_theta1 - (learning_rate * d_theta * 1/float(len(data)))

    return new_theta0, new_theta1


# updates parameters theta0, theta1
def gradient_descent(data, no_of_iterations, learning_rate, initial_theta0, initial_theta1):

    # initialize hypothesis variables
    theta0 = initial_theta0
    theta1 = initial_theta1

    # begin learning
    for i in range(no_of_iterations):
        # update
        theta0, theta1 = gradient_step(data, learning_rate, theta0, theta1)

    # return learned values
    return theta0, theta1


# initialize data and parameters then call gradient_descent
def run():

    # load data set
    data = np.genfromtxt('data.csv', delimiter=',', names=['x', 'y'])

    # learning rate for the algorithm
    learning_rate = 0.0001

    # iteration period
    no_of_iterations = 1000

    # initial values for theta0, theta1 for the hypothesis function
    # h(x) = theta0 + theta1*x
    initial_theta0 = 0
    initial_theta1 = 0

    # run gradient descent to generate the hypothesis function
    theta0, theta1 = gradient_descent(data, no_of_iterations, learning_rate, initial_theta0, initial_theta1)

    # plot data set
    plt.scatter(data['x'], data['y'])

    # Add hypothesis function line
    x = np.array([0, 100])
    plt.plot(x, theta0 + theta1 * x)

    plt.show()


if __name__ == "__main__":
    run()
