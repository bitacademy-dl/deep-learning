# nand gate: perceptron
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Found')


def NAND(x):
    w, b = np.array([-0.5, -0.5]), 0.7

    a = np.sum(x * w) + b
    #y = 1 if a > 0 else 0
    y = step(a)

    return y


if __name__ == '__main__':
    y1 = NAND(np.array([0, 0]))
    print(y1)

    y2 = NAND(np.array([0, 1]))
    print(y2)

    y3 = NAND(np.array([1, 0]))
    print(y3)

    y4 = NAND(np.array([1, 1]))
    print(y4)