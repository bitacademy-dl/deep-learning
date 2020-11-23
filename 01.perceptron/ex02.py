# and gate: perceptron
import numpy as np


def AND(x):
    w, b = np.array([0.5, 0.5]), np.array(-0.7)
    a = np.sum(x * w) + b

    if a < 0:
        return 0
    else:
        return 1


y1 = AND(np.array([0, 0]))
print(y1)

y2 = AND(np.array([0, 1]))
print(y2)

y3 = AND(np.array([1, 0]))
print(y3)

y4 = AND(np.array([1, 1]))
print(y4)
