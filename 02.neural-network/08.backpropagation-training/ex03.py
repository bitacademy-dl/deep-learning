# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: MultiLayerNet
# Test: Backpropagation Gradient vs Numerical Gradient
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2.initialize network
network.initialize(input_size=train_x.shape[1], hidden_size=[50], output_size=train_t.shape[1])

# 3. batch by 3
train_x_batch = train_x[:3]
train_t_batch = train_t[:3]

# 4. gradient
gradient_numerical = network.numerical_gradient_net(train_x_batch, train_t_batch)
gradient_backpropagation = network.backpropagation_gradient_net(train_x_batch, train_t_batch)

# 5. mean of modulus
for key in gradient_numerical:
    diff = np.average(np.abs((gradient_numerical[key] - gradient_backpropagation[key])))
    print(f'{key} difference: {diff}')

# 6. 결론: 거의 차이 없음!
# w1 difference: 4.4934869968055257e-10
# b1 difference: 2.215194909444126e-09
# w2 difference: 5.393343686911959e-09
# b2 difference: 1.3999702521694247e-07
