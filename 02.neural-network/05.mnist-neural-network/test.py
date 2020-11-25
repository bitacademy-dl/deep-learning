import numpy as np


a1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a2 = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 20, 30, 40, 50, 60, 70, 80, 90]
])

b1 = np.max(a1)
b2 = np.max(a2)
print(b1, b2)
