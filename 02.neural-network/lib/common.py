import numpy as np


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# relu activation function
def relu(x):
    # if x > 0:
    #     return x
    # else:
    #     return 0
    # return x if x > 0 else 0
    return np.maximum(0, x)
