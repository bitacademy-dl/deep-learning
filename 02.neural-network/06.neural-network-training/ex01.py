# 신경망학습: 오차제곱합 손실함수(Sum of Squares Error, SSE)
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sum_squares_error
except ImportError:
    print('Library Module Can Not Found')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.])
y2 = np.array([0.1, 0.05, 0., 0.4, 0.02, 0.03, 0.1, 0.3, 0., 0.])
y3 = np.array([0., 0.92, 0.02, 0., 0.02, 0.03, 0.01, 0., 0., 0.])

# print(np.sum(y1))
# print(np.sum(y2))
# print(np.sum(y3))

# test
print(sum_squares_error(y1, t))
print(sum_squares_error(y2, t))
print(sum_squares_error(y3, t))

