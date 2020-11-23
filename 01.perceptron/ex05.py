# multi layer perceptron

import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import identity
    from ex02 import AND
    from ex03 import NAND
    from ex04 import OR
except ImportError:
    print('Library Module Can Not Found')


def XOR(x):
    a1 = NAND(x)
    a2 = OR(x)
    a3 = AND(np.array([a1, a2]))

    y = identity(a3)

    return y


y1 = XOR(np.array([0, 0]))
print(y1)

y2 = XOR(np.array([0, 1]))
print(y2)

y3 = XOR(np.array([1, 0]))
print(y3)

y4 = XOR(np.array([1, 1]))
print(y4)