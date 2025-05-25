import numpy as np


# sigmoid function
# each neuron receives inputs, applies weights and bias,
# and then passes the result through an activation function like sigmoid.
# sigmoid func always outputs a value b/w 0 and 1.
# o(z) = 1/(1 + e^-z)
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
# 3x4 matrix
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# random numbers to make calcs deterministic
np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))  # dot product of vectors

    # how much did we miss
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output after training")
print(l1)
