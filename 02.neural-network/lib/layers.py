# Sigmoid Layer
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error
except ImportError:
    print('Library Module Can Not Found')


# SoftmaxWithLoss Layer
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t=None):
        self.y = softmax(x)
        self.t = t

        if self.t is None:
            return self.y

        # 역전파시 전치행렬에 대한 내적을 대비
        if self.t.ndim == 1:
            self.t = self.t[np.newaxis, :]

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, x):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) / batch_size
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out
        return dx

# ReLU Layer
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0

        dx = dout
        return dx

# ===============================

# Multilply Layer
class Multiply:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy


# Add Layer
class Add:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

