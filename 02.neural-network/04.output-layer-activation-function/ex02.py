# 출력함수(출력층 활성함수) σ() - 소프트맥스함수(softmax Function)
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, softmax_overflow
except ImportError:
    print('Library Module Can Not Found')

# test1
a = np.array([0.3, 1., 0.78])
y = softmax(a)
print(y, np.sum(y))

# test2: 큰값: 800.
# a = np.array([0.3, 800., 0.78])
# y = softmax_overflow(a)
# print(y)

# test2: 큰값: 800.
a = np.array([0.3, 800., 0.78])
y = softmax(a)
print(y, np.sum(y))

