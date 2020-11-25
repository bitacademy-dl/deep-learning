# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, numerical_gradient1
except ImportError:
    print('Library Module Can Not Found')


def loss(w, x, t):
    a = np.dot(x, w)
    y = softmax(a)
    e = cross_entropy_error(y, t)

    return e


_x = np.array([0.6, 0.9])   # 입력(x)          2 vector
_t = np.array([0., 0., 1.]) # label(one-hot)  3 vector
_w = np.random.randn(2, 3)  # weight          2 x 3 matrix

g = numerical_gradient1(loss, _w, _x, _t)




