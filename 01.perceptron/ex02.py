# and gate: perceptron
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Found')


def AND(x):
    w, b = np.array([0.5, 0.5]), np.array(-0.7)

    a = np.sum(x * w) + b
    y = step(a)

    return y


y1 = AND(np.array([0, 0]))
print(y1)

y2 = AND(np.array([0, 1]))
print(y2)

y3 = AND(np.array([1, 0]))
print(y3)

y4 = AND(np.array([1, 1]))
print(y4)
