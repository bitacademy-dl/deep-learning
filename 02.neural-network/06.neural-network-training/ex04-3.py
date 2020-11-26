# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, numerical_gradient2
except ImportError:
    print('Library Module Can Not Found')

x = np.array([0.6, 0.9])       # 입력(x)          2 vector
t = np.array([0., 0., 1.])     # label(one-hot)  3 vector


def forward_progation(w):
    a = np.dot(x, w)
    y = softmax(a)
    return y


def loss(w):
    y = forward_progation(w)
    e = cross_entropy_error(y, t)
    return e


_w = np.array([
    [0.02, 0.224, 0.135],
    [0.01, 0.052, 0.345]
])                              # weight          2 x 3 matrix

g = numerical_gradient2(loss, _w)
print(g)



