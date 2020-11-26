import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')


# 1.load training/test data
x = np.array([
    [0.6, 0.9, 0.11],
    [0.6, 0.9, 0.11],
    [0.6, 0.9, 0.11]
])                          # 입력(x)          3 x 3 matrox
t = np.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [0.6, 0.9, 0.11]
])                          # label(one-hot)  3 x 3 matrix

# 2.hyperparameters
numiters = 1  # 10000
szbatch = 100

# 3. initialize network
network.initialize(sz_input=3, sz_hidden=3, sz_output=3)


gradient = network.numerical_gradient_net(x, t)
print(gradient)
