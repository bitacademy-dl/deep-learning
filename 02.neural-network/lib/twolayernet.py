# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, sigmoid, cross_entropy_error
except ImportError:
    print('Library Module Can Not Found')

params = dict()


def initialize(sz_input, sz_hidden, sz_output, w_init=0.01):
    params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)
    params['b1'] = np.zeros(sz_hidden)
    params['w2'] = w_init * np.random.randn(sz_hidden, sz_output)
    params['b2'] = np.zeros(sz_output)


def forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = sigmoid(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)

    return y


def loss(x, t):
    y = forward_propagation(x)
    e = cross_entropy_error(y, t)
    return e


def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss(x, t)

            param[idx] = temp - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = temp   # 값복원
            it.iternext()

        gradient[key] = param_gradient

    return gradient




