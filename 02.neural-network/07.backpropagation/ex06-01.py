# Sigmoid Layer Test
import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Affine, ReLU, SoftmaxWithLoss
    from common import softmax
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')


def backpropagation_gradient_net(x, t):
    forward_propagation(x, t)
    backward_propagation(1)

    idxaffine = 0
    gradient = dict()

    for layer in _layers:
        if type(layer).__name__ == 'Affine':
            idxaffine += 1
            gradient[f'w{idxaffine}'] = layer.dw
            gradient[f'b{idxaffine}'] = layer.db

    return gradient


def backward_propagation(dout):
    for layer in _layers[::-1]:
        dout = layer.backward(dout)

    return dout


def forward_propagation(x, t=None):
    for layer in _layers:
        x = layer.forward(x, t) if type(layer).__name__ == 'SoftmaxWithLoss' and t is not None else layer.forward(x)

    return x


def loss(x, t):
    y = forward_propagation(x, t)
    return y


network.forward_propagation = forward_propagation
network.loss = loss

# ========================================================

# 1.load training/test data
_x, _t = np.array([2.6, 3.9, 5.6]), np.array([0, 0, 1])

# 2.hyperparameter


# = Numerical Gradient ======================================================

# 3.initialize network
network.initialize(3, 2, 3)

_layers = [
    Affine(network.params['w1'], network.params['b1']),
    ReLU(),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

grad = network.numerical_gradient_net(_x, _t)
print(grad)

# = backpropagation gradient ======================================================
# 3.initialize network
network.initialize(3, 2, 3)

_layers = [
    Affine(network.params['w1'], network.params['b1']),
    ReLU(),
    Affine(network.params['w2'], network.params['b2']),
    SoftmaxWithLoss()
]

grad = backpropagation_gradient_net(_x, _t)
print(grad)

