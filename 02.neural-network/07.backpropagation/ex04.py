# Sigmoid Layer Test
import os
import sys
from pathlib import Path

import numpy as np

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Sigmoid
except ImportError:
    print('Library Module Can Not Found')

# Test1(Vector)
layer = Sigmoid()

x = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
print(x)

y = layer.forward(x)
print(y)
print(layer.out)

dout = np.array([-0.1, -0.2, -0.3, 0.4, -0.5])
dout = layer.backward(dout)
print(dout)

print('=========================================')

# Test2(Matrix)
x = np.array([
    [0.1, -0.5, 1.0],
    [0.2, -0.6, 2.0],
    [0.3, -0.7, 3.0],
    [0.4, -0.8, 4.0]
])
y = layer.forward(x)
print(y)
print(layer.out)

dout = np.array([
    [-0.1, 10.5, -1.0],
    [-0.2, 11.6, -2.0],
    [-0.3, 9.7, -3.0],
    [-0.4, 8.8, -4.0]
])
dout = layer.backward(dout)
print(dout)
